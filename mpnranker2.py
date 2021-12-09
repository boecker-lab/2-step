# from tensorboardX.writer import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from utils import BatchGenerator, Data
import numpy as np
from pprint import pprint
import logging

logger = logging.getLogger('rtranknet.mpnranker2')
info = logger.info
warning = logger.warning

class MPNranker(nn.Module):
    def __init__(self, extra_features_dim=0, sys_features_dim=0,
                 hidden_units=[16, 8], hidden_units_pv=[16, 2], encoder_size=300,
                 depth=3, dropout_rate=0.0):
        super(MPNranker, self).__init__()
        args = TrainArgs()
        args.from_dict({'dataset_type': 'classification',
                        'data_path': None,
                        'hidden_size': encoder_size,
                        'depth': depth,
                        'dropout': dropout_rate})
        self.encoder = MPNEncoder(args, get_atom_fdim(), get_bond_fdim())
        self.extra_features_dim = extra_features_dim
        self.sys_features_dim = sys_features_dim
        self.encextra_size = self.encoder.hidden_size + extra_features_dim + sys_features_dim
        # System x molecule preference encoding
        self.hidden_pv = nn.ModuleList()
        for i, u in enumerate(hidden_units_pv):
            self.hidden_pv.append(Linear(self.encextra_size
                                         if i == 0 else hidden_units_pv[i - 1], u))
        # actual ranking layers
        self.hidden = nn.ModuleList()
        for i, u in enumerate(hidden_units):
            self.hidden.append(Linear(self.encoder.hidden_size + extra_features_dim + hidden_units_pv[-1]
                                      if i == 0 else hidden_units[i - 1], u))
        # One ROI value for (graph,extr,sys) sample
        self.ident = Linear(hidden_units[-1], 1)
    def forward(self, batch):
        """(1|2|n) x [batch_size x (smiles|graphs), batch_size x extra_features, batch_size x sys_features]"""
        res = []
        for graphs, extra, sysf in batch:       # normally 1 or 2
            # encode molecules
            if (isinstance(graphs[0], str)):
                graphs = [mol2graph([_]) for _ in graphs]
            enc = torch.cat([self.encoder(g) for g in graphs], 0)  # [batch_size x encoder size]
            # encode system x molecule relationships
            enc_pv = torch.cat([enc, extra, sysf], 1)
            for h in self.hidden_pv:
                enc_pv = F.relu(h(enc_pv))
            # now ranking layers: [enc, enc_pv] -> ROI
            enc = torch.cat([enc, extra, enc_pv], 1)
            for h in self.hidden:
                enc = F.relu(h(enc))
            # single ROI value
            roi = self.ident(enc)                          # TODO: relu?
            res.append(roi.transpose(0, 1)[0])      # [batch_size]
        if (len(res) > 2):
            raise Exception('only one or two molecules are supported for now, not ', len(res))
        # return torch.sigmoid(res[0] - res[1] if len(res) == 2 else res[0])
        return [torch.sigmoid(r) for r in res]

    def predict(self, graphs, extra, sysf, batch_size=8192,
                prog_bar=False, ret_features=False):
        self.eval()
        preds = []
        features = []
        if (not isinstance(extra, torch.Tensor)):
            extra = torch.as_tensor(extra).float().to(self.encoder.device)
        if (not isinstance(sysf, torch.Tensor)):
            sysf = torch.as_tensor(sysf).float().to(self.encoder.device)
        it = range(np.ceil(len(graphs) / batch_size).astype(int))
        if (prog_bar):
            it = tqdm(it)
        with torch.no_grad():
            for i in it:
                start = i * batch_size
                end = i * batch_size + batch_size
                batch = (graphs[start:end], extra[start:end],
                         sysf[start:end])
                preds.append(self((batch, ))[0].cpu().detach().numpy())
                if (ret_features):
                    features.extend([self.encoder(g) for g in graphs[start:end]])
        if (ret_features):
            return np.concatenate(preds), np.concatenate(features)
        return np.concatenate(preds)
    def loss_step(self, x, y, weights, loss_fun):
        pred = self(x)
        y = torch.as_tensor(y).float().to(self.encoder.device)
        weights = torch.as_tensor(weights).to(self.encoder.device)
        if isinstance(loss_fun, nn.MarginRankingLoss):
            loss = ((loss_fun(*pred, y) * weights).mean(), loss_fun(*pred, y) * weights)
        else:
            loss = ((loss_fun(pred, y) * weights).mean(), loss_fun(pred, y) * weights)
        return loss

def data_predict(ranker: MPNranker, data: Data, batch_size=8192):
    preds = {}
    for ds in data.df.dataset_id.unique():
        indices = [data.df.index.get_loc(i) for i in
                   data.df.loc[data.df.dataset_id == ds].index]
        mols = data.df['smiles.std'].iloc[indices].tolist()
        extraf = data.x_features[indices]
        sysf = data.x_info[indices]
        preds[ds] = (ranker.predict(mols, extraf, sysf, batch_size=batch_size),
                     data.get_y()[indices])
    return preds

def data_eval(ranker: MPNranker, data: Data, batch_size=8192,
              epsilon=0.5):
    """returns ds_weighted_acc, mean_acc, {ds: acc}
    """
    preds = data_predict(ranker, data, batch_size)
    accs = {}
    for ds, (y_pred, y) in preds.items():
        accs[ds] = eval_(y, y_pred, epsilon=epsilon)
    total_num = sum(len(v[0]) for v in preds.values())
    return (sum(accs[ds] * len(preds[ds][0]) for ds in accs) / total_num,
            sum(accs.values()) / len(accs), accs)


class RankerTwins(nn.Module):
    def __init__(self, **ranker_args):
        super(RankerTwins, self).__init__()
        self.ranker = MPNranker(**ranker_args)
        self.confl_hidden0 = nn.Linear(2, 8)
        self.confl_hidden1 = nn.Linear(8, 1)
        self.confl_rank = nn.Sigmoid()
    def forward(self, batch):
        batch1, batch2 = batch
        y1, z1 = self.ranker(batch1)
        y2, z2 = self.ranker(batch2)
        # return y1, y2, self.confl_rank(torch.sum(torch.stack([*z1, *z2], axis=1), axis=1))
        # return y1, y2, self.confl_rank(
        #     self.confl_hidden1(self.confl_hidden0(torch.stack([z1[0] + z1[1], z2[0] + z2[1]], axis=1)))).transpose(0, 1)[0]
        return y1, y2, self.confl_rank(z1[0] + z1[1] - z2[0] + z2[1])
            # self.confl_hidden1(self.confl_hidden0(torch.stack([z1[0] + z1[1], z2[0] + z2[1]], axis=1)))).transpose(0, 1)[0]

    def loss_step(self, x1, x2, y1, y2, p1weights, p2weights, yconfl,
                  conflweights, loss_fun_rank, loss_fun_twin):
        y1_pred, y2_pred, yconfl_pred = self((x1, x2))
        y1, y2, yconfl,  p1weights, p2weights, conflweights = [
            torch.as_tensor(_).float().to(self.ranker.encoder.device)
            for _ in [y1, y2, yconfl, p1weights, p2weights, conflweights]]
        # loss = (rank_loss1 + rank_loss2) / 2 + confl_mod * confl_loss
        loss1 = ((
            loss_fun_rank(y1_pred, y1) * p1weights
            + loss_fun_rank(y2_pred, y2) * p2weights) / 2).mean()
        loss2 = (loss_fun_twin(yconfl_pred, yconfl) * conflweights).mean()
        return loss1, loss2

def twin_train(twins: RankerTwins, epochs: int,
               train_loader:DataLoader, val_loader:DataLoader=None,
               writer:SummaryWriter=None, val_writer:SummaryWriter=None,
               confl_mod=0.5, learning_rate=1e-4, steps_train_loss=10,
               steps_val_loss=100, ep_save=False):
    save_name = ('rankertwins' if writer is None else
                 writer.get_logdir().split('/')[-1].replace('_train', ''))
    optimizer = torch.optim.Adam(twins.parameters(), lr=learning_rate)
    rank_loss_fun = nn.BCELoss(reduction='none')
    twins_loss_fun = nn.BCELoss(reduction='none')
    twins.train()
    global_iter_count = 0
    for epoch in range(epochs):
        iter_count = val_iter_count = 0
        loss_sum = val_loss_sum = 0
        rank_loss_sum = val_rank_loss_sum = 0
        confl_loss_sum = val_confl_loss_sum = 0
        loop = tqdm(train_loader)
        for pairs in loop:
            twins.zero_grad()
            loss_rank, loss_confl = twins.loss_step(
                (pairs.p1.x1, pairs.p1.x2), (pairs.p2.x1, pairs.p2.x2),
                pairs.p1.y, pairs.p2.y, pairs.p1.weights, pairs.p2.weights,
                pairs.confl, pairs.weights, rank_loss_fun, twins_loss_fun)
            loss = (1 - confl_mod) * loss_rank + confl_mod * loss_confl
            loss_sum += loss.item()
            rank_loss_sum += loss_rank.item()
            confl_loss_sum += loss_confl.item()
            iter_count += 1
            global_iter_count += 1
            loss.backward()
            optimizer.step()
            if (iter_count % steps_train_loss == (steps_train_loss - 1) and writer is not None):
                writer.add_scalar('loss', loss_sum / iter_count, global_iter_count)
                writer.add_scalar('rank_loss', rank_loss_sum / iter_count, global_iter_count)
                writer.add_scalar('confl_loss', confl_loss_sum / iter_count, global_iter_count)
            if (val_writer is not None and val_loader is not None and len(val_loader) > 0
                and iter_count % steps_val_loss == (steps_val_loss - 1)):
                twins.eval()
                with torch.no_grad():
                    for pairs in val_loader:
                        loss_rank, loss_confl = twins.loss_step(
                            (pairs.p1.x1, pairs.p1.x2), (pairs.p2.x1, pairs.p2.x2),
                            pairs.p1.y, pairs.p2.y, pairs.p1.weights, pairs.p2.weights,
                            pairs.confl, pairs.weights, rank_loss_fun, twins_loss_fun)
                        loss = (1 - confl_mod) * loss_rank + confl_mod * loss_confl
                        val_loss_sum += loss.item()
                        val_rank_loss_sum += loss_rank.item()
                        val_confl_loss_sum += loss_confl.item()
                val_writer.add_scalar('loss', val_loss_sum / val_iter_count, global_iter_count)
                val_writer.add_scalar('rank_loss', val_rank_loss_sum / val_iter_count, global_iter_count)
                val_writer.add_scalar('confl_loss', val_confl_loss_sum / val_iter_count, global_iter_count)
                val_writer.flush()
                writer.flush()
                twins.train()
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(loss=loss_sum/iter_count if iter_count > 0 else np.infty,
                             rank_loss=rank_loss_sum/iter_count if iter_count > 0 else np.infty,
                             confl_loss=confl_loss_sum/iter_count if iter_count > 0 else np.infty,
                             val_loss=val_loss_sum/val_iter_count if val_iter_count > 0 else np.infty)
            if (ep_save):
                torch.save(twins, f'{save_name}_ep{epoch + 1}.pt')

def train(ranker: MPNranker, bg: Union[BatchGenerator, DataLoader], epochs=2,
          writer:SummaryWriter=None, val_g: Union[BatchGenerator, DataLoader]=None,
          epsilon=0.5, val_writer:SummaryWriter=None,
          confl_writer:SummaryWriter=None,
          steps_train_loss=10, steps_val_loss=100,
          batch_size=8192, sigmoid_loss=False,
          margin_loss=0.1, early_stopping_patience=None,
          ep_save=False, learning_rate=1e-3, no_encoder_train=False,
          accs=True):
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
    # loss_fun = nn.BCEWithLogitsLoss(reduction='none')
    # loss_fun = nn.BCELoss(reduction='none')
    loss_fun = nn.MarginRankingLoss(margin_loss, reduction='none')
    if (sigmoid_loss):
        warning('sigmoid loss is not implemented anymore! margin loss will be used')
    ranker.train()
    loss_sum = iter_count = val_loss_sum = val_iter_count = val_pat = 0
    confl_loss = {}
    rel_confl_len = 0
    confl_loss_avg = np.nan
    last_val_step = np.infty
    stop = False
    for epoch in range(epochs):
        if stop:                # CTRL+C
            break
        loop = tqdm(bg)
        for x, y, weights in loop:
            if (ranker.extra_features_dim > 0):
                x[0][1] = torch.as_tensor(x[0][1]).float().to(ranker.encoder.device)
                x[1][1] = torch.as_tensor(x[1][1]).float().to(ranker.encoder.device)
            if (ranker.sys_features_dim > 0):
                x[0][2] = torch.as_tensor(x[0][2]).float().to(ranker.encoder.device)
                x[1][2] = torch.as_tensor(x[1][2]).float().to(ranker.encoder.device)
            ranker.zero_grad()
            loss = ranker.loss_step(x, y, weights, loss_fun)
            loss_sum += loss[0].item()
            iter_count += 1
            loss[0].backward()
            if (confl_writer is not None):
                confl_weight_thr = 0.9 * bg.dataset.confl_weight
                if ((weights > confl_weight_thr).any()): # TODO: DEBUG for confl pairs
                    # import pdb; pdb.set_trace()
                    # NOTE: makes the following assumptions:
                    # 1. confl_weights_modifier > 9 and groups roughly balanced so that
                    # all confl pairs have weights >9
                    # 2. has extra features from which the first one is used as ID for the
                    # compound (logp, 5 decimals)
                    for g1, g2, yi, l in zip(np.array(x[0][0])[weights > confl_weight_thr],
                                             np.array(x[1][0])[weights > confl_weight_thr],
                                             y[weights > confl_weight_thr],
                                             loss[1][weights > confl_weight_thr] / weights[weights > confl_weight_thr]):
                        # print(g1, g2)
                        # print(id(g1), id(g2))
                        confl_loss.setdefault(frozenset((id(g1), id(g2))), {}
                                              )[yi.item()] = l.item()
                    # pprint(confl_loss)
                    rel_confl_items = [(v[0 if sigmoid_loss else -1] + v[1]) / 2
                                       for k, v in confl_loss.items() if 1 in v and (0 if sigmoid_loss else -1) in v]
                    # pprint(rel_confl_items)
                    if (len(rel_confl_items) > 0 and  len(rel_confl_items) >= rel_confl_len):
                        # print(rel_confl_items)
                        # a new conflicting pair was trained on with conflicting target values
                        rel_confl_len = len(rel_confl_items)
                        confl_loss_avg = sum(rel_confl_items) / rel_confl_len
                        if (confl_writer is not None):
                            confl_writer.add_scalar('loss', confl_loss_avg, iter_count)
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
                        if (ranker.sys_features_dim > 0):
                            x[0][2] = torch.as_tensor(x[0][2]).float().to(ranker.encoder.device)
                            x[1][2] = torch.as_tensor(x[1][2]).float().to(ranker.encoder.device)
                        val_loss_sum += ranker.loss_step(x, y, weights, loss_fun)[0].item()
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
                             val_loss=val_loss_sum/val_iter_count if val_iter_count > 0 else np.nan,
                             confl_loss=confl_loss_avg)
        if val_writer is not None:
            val_writer.flush()
        ranker.eval()
        if accs and writer is not None:
            train_acc = eval_(bg.dataset.y, ranker.predict(
                bg.dataset.x_mols, bg.dataset.x_extra, bg.dataset.x_sys, batch_size=batch_size), epsilon=epsilon)
            writer.add_scalar('acc', train_acc, iter_count)
            writer.flush()
            print(f'{train_acc=:.2%}')
            if (val_writer is not None):
                if (val_g.dataset.dataset_info is not None):
                    val_accs = []
                    for ds in set(val_g.dataset.dataset_info):
                        ds_indices = [i for i, dsi in enumerate(val_g.dataset.dataset_info) if dsi == ds]
                        val_acc = eval_(val_g.dataset.y[ds_indices], ranker.predict(
                            val_g.dataset.x_mols[ds_indices], val_g.dataset.x_extra[ds_indices],
                            val_g.dataset.x_sys[ds_indices], batch_size=batch_size), epsilon=epsilon)
                        if (not np.isnan(val_acc)):
                            val_accs.append(val_acc)
                        print(f'{ds}: \t{val_acc=:.2%}')
                    val_acc = np.mean(val_accs)
                else:
                    val_acc = eval_(val_g.dataset.y, ranker.predict(val_g.dataset.x_mols, val_g.dataset.x_extra, val_g.dataset.x_sys,
                                                            batch_size=batch_size), epsilon=epsilon)
                val_writer.add_scalar('acc', val_acc, iter_count)
                val_writer.flush()
                print(f'{val_acc=:.2%}')

        if (ep_save):
            torch.save(ranker, f'{save_name}_ep{epoch + 1}.pt')
        ranker.train()
