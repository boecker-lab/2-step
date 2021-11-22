# from tensorboardX.writer import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from typing import List, Union, Tuple
from tqdm import tqdm
from torch.nn.modules.linear import Linear
from chemprop.models.mpn import (MPNEncoder, get_atom_fdim, get_bond_fdim)
from chemprop.args import TrainArgs
from chemprop.features import mol2graph, BatchMolGraph
from evaluate import eval_
from utils import BatchGenerator
import numpy as np
from pprint import pprint
import logging


class MPNranker(nn.Module):
    def __init__(self, sigmoid=False, extra_features_dim=0,
                 sys_features_dim=0,
                 hidden_units=[],
                 encoder_size=300, dropout_rate=0.0):
        super(MPNranker, self).__init__()
        self.sigmoid = sigmoid
        args = TrainArgs()
        args.from_dict({'dataset_type': 'classification',
                        'data_path': None,
                        'hidden_size': encoder_size,
                        'dropout': dropout_rate})
        self.encoder = MPNEncoder(args, get_atom_fdim(), get_bond_fdim())
        self.extra_features_dim = extra_features_dim
        self.sys_features_dim = sys_features_dim
        self.encextra_size = self.encoder.hidden_size + extra_features_dim
        self.features_ident = Linear(self.encextra_size, 1)
        self.hidden = nn.ModuleList()
        for i, u in enumerate(hidden_units):
            self.hidden.append(Linear(1 + sys_features_dim
                                      if i == 0 else hidden_units[i - 1], u))
        self.res_ident = Linear(1 + sys_features_dim
                                if len(hidden_units) == 0 else hidden_units[-1], 1)
        self.rank = nn.Sigmoid()
    def forward(self, batch, features=False):
        res = []
        (graphs1, extra1), (graphs2, extra2), sys = batch
        # for each compound
        for graphs, extra in [(graphs1, extra1), (graphs2, extra2)]:
            # encode molecules
            if (isinstance(graphs[0], str)):
                graphs = [mol2graph([_]) for _ in graphs]
            enc = torch.cat([self.encoder(g) for g in graphs], 0)  # [batch_size x 300]
            # include extra compound fetures (e.g., MolLogP)
            enc = torch.cat([enc, extra], 1)
            res.append(self.features_ident(enc).transpose(0, 1)[0])      # [batch_size]
        # for the whole pair
        assert len(res) == 2
        pair = res[1] - res[0]
        concat = torch.cat([torch.stack([pair]).transpose(0,1), sys], 1) if sys is not None else pair
        for h in self.hidden:
            concat = h(concat)
        # single value
        return(self.rank(self.res_ident(concat)).transpose(0, 1)[0])
        if (features):
            return res
        if (self.sigmoid and len(res) == 2):
            return self.rank(res[0] - res[1]) # batch_size
        else:
            return res

def predict(x, ranker: MPNranker, batch_size=8192,
            prog_bar=False, ret_features=False):
    ranker.eval()
    preds = []
    features = []
    if ranker.extra_features_dim > 0:
        graphs, extra = x
        extra = torch.as_tensor(extra).float().to(ranker.encoder.device)
    else:
        graphs, extra = (x, None)
    it = range(np.ceil(len(graphs) / batch_size).astype(int))
    if (prog_bar):
        it = tqdm(it)
    with torch.no_grad():
        for i in it:
            start = i * batch_size
            end = i * batch_size + batch_size
            batch = ((graphs[start:end], extra[start:end]) if extra is not None
                     else graphs[start:end])
            preds.append(ranker((batch, ))[0].cpu().detach().numpy())
            if (ret_features):
                features.extend([ranker.encoder(g) for g in graphs[start:end]])
    if (ret_features):
        return np.concatenate(preds), np.concatenate(features)
    return np.concatenate(preds)

def loss_step(ranker, x, y, weights, loss_fun):
    pred = ranker(x)
    y = torch.as_tensor(y).float().to(ranker.encoder.device)
    weights = torch.as_tensor(weights).to(ranker.encoder.device)
    if isinstance(loss_fun, nn.MarginRankingLoss):
        loss = ((loss_fun(*pred, y) * weights).mean(), loss_fun(*pred, y) * weights)
    else:
        loss = ((loss_fun(pred, y) * weights).mean(), loss_fun(pred, y) * weights)
    return loss

def train(ranker: MPNranker, bg: Union[BatchGenerator, DataLoader], epochs=2,
          writer:SummaryWriter=None, val_g: Union[BatchGenerator, DataLoader]=None,
          epsilon=0.5, val_writer:SummaryWriter=None,
          confl_writer:SummaryWriter=None,
          steps_train_loss=10, steps_val_loss=100,
          batch_size=8192, sigmoid_loss=False,
          margin_loss=0.1, early_stopping_patience=None,
          ep_save=False, learning_rate=1e-3, no_encoder_train=False,
          accs=False):
    save_name = ('mpnranker' if writer is None else
                 writer.get_logdir().split('/')[-1].replace('_train', ''))
    ranker.to(ranker.encoder.device)
    print('device:', ranker.encoder.device)
    if (no_encoder_train):
        for p in ranker.encoder.parameters():
            p.requires_grad = False
    optimizer = optim.Adam(ranker.parameters(), lr=learning_rate)
    # loss_fun = (nn.BCELoss(reduction='none') if sigmoid_loss
    #             else nn.MarginRankingLoss(margin_loss, reduction='none'))
    loss_fun = nn.BCEWithLogitsLoss(reduction='none')
    ranker.train()
    loss_sum = iter_count = val_loss_sum = val_iter_count = val_pat = 0
    confl_loss = {}
    rel_confl_len = 0
    last_val_step = np.infty
    stop = False
    for epoch in range(epochs):
        if stop:
            break
        loop = tqdm(bg)
        for x, y, weights in loop:
            ranker.zero_grad()
            loss = loss_step(ranker, x, y, weights, loss_fun)
            loss_sum += loss[0].item()
            iter_count += 1
            loss[0].backward()
            if (confl_writer is not None):
                if ((weights > 9).any()): # TODO: DEBUG for confl pairs
                    # NOTE: makes the following assumptions:
                    # 1. confl_weights_modifier > 9 and groups roughly balanced so that
                    # all confl pairs have weights >9
                    # 2. has extra features from which the first one is used as ID for the
                    # compound (logp, 5 decimals)
                    for logp1, logp2, yi, l in zip(x[0][1][weights > 9, 0], x[1][1][weights > 9, 0],
                                                   y[weights > 9], loss[1][weights > 9]):
                        logp1, logp2 = f'{logp1:.5f}', f'{logp2:.5f}'
                        confl_loss.setdefault(frozenset([logp1, logp2]), {})[yi] = l.item()
                    rel_confl_items = [(v[0 if sigmoid_loss else -1] + v[1]) / 2
                                       for k, v in confl_loss.items() if 1 in v and (0 if sigmoid_loss else -1) in v]
                    if (len(rel_confl_items) > rel_confl_len):
                        # a new conflicting pair was trained on with conflicting target values
                        rel_confl_len = len(rel_confl_items)
                        if (confl_writer is not None):
                            confl_writer.add_scalar('loss', sum(rel_confl_items) / rel_confl_len, iter_count)
                    # pprint(ranker.get_parameter('hidden.0.weight').grad[:, -ranker.extra_features_dim:])
                    # pprint(ranker.get_parameter('hidden.0.weight').grad[:, -ranker.extra_features_dim+1:].sum(1))
                    # pprint([(p[0], p[1].size(), p[1].grad) for p in ranker.named_parameters()])
            optimizer.step()
            if (iter_count % steps_train_loss == (steps_train_loss - 1) and writer is not None):
                loss_avg = loss_sum / iter_count
                if writer is not None:
                    writer.add_scalar('loss', loss_avg, iter_count)
                else:
                    print(f'Loss = {loss_avg:.4f}')
            if (val_writer is not None and len(val_g) > 0
                and iter_count % steps_val_loss == (steps_val_loss - 1)):
                ranker.eval()
                with torch.no_grad():
                    for x, y, weights in val_g:
                        if (ranker.extra_features_dim > 0):
                            x[0][1] = torch.as_tensor(x[0][1]).float().to(ranker.encoder.device)
                            x[1][1] = torch.as_tensor(x[1][1]).float().to(ranker.encoder.device)
                        val_loss_sum += loss_step(ranker, x, y, weights, loss_fun)[0].item()
                        val_iter_count += 1
                val_step = val_loss_sum / val_iter_count
                val_writer.add_scalar('loss', val_step, iter_count)
                if (early_stopping_patience is not None and val_step > last_val_step):
                    if (val_pat >= early_stopping_patience):
                        print(f'early stopping; patience_count={val_pat}, {val_step=} > {last_val_step=}')
                        stop = True
                        break
                    val_pat += 1
                last_val_step = min(val_step, last_val_step)
                ranker.train()
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(loss=loss_sum/iter_count if iter_count > 0 else np.nan,
                             val_loss=val_loss_sum/val_iter_count if val_iter_count > 0 else np.nan)
        val_writer.flush()
        ranker.eval()
        if accs and writer is not None:
            train_acc = eval_(bg.y, predict(bg.x, ranker, batch_size=batch_size), epsilon=epsilon)
            writer.add_scalar('acc', train_acc, iter_count)
            writer.flush()
            print(f'{train_acc=:.2%}')
            if (val_writer is not None):
                if (val_g.dataset_info is not None):
                    val_accs = []
                    for ds in set(val_g.dataset_info):
                        if (val_g.multix):
                            x = [[val_g.x[j][i] for i in range(len(val_g.dataset_info)) if val_g.dataset_info[i] == ds]
                                 for j in range(len(val_g.x))]
                        else:
                            x = [val_g.x[i] for i in range(len(val_g.dataset_info)) if val_g.dataset_info[i] == ds]
                        y = [val_g.y[i] for i in range(len(val_g.dataset_info)) if val_g.dataset_info[i] == ds]
                        val_acc = eval_(y, predict(x, ranker, batch_size=batch_size), epsilon=epsilon)
                        if (not np.isnan(val_acc)):
                            val_accs.append(val_acc)
                        print(f'{ds}: \t{val_acc=:.2%}')
                    val_acc = np.mean(val_accs)
                else:
                    val_acc = eval_(val_g.y, predict(val_g.x, ranker, batch_size=batch_size), epsilon=epsilon)
                val_writer.add_scalar('acc', val_acc, iter_count)
                val_writer.flush()
                print(f'{val_acc=:.2%}')

        if (ep_save):
            torch.save(ranker, f'{save_name}_ep{epoch + 1}.pt')
        ranker.train()
