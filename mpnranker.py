from tensorboardX.writer import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
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


class MPNranker(nn.Module):
    def __init__(self, sigmoid=False, extra_features_dim=0, hidden_units=[]):
        super(MPNranker, self).__init__()
        self.sigmoid = sigmoid
        args = TrainArgs()
        args.from_dict({'dataset_type': 'classification',
                        'data_path': None})
        self.encoder = MPNEncoder(args, get_atom_fdim(), get_bond_fdim())
        self.extra_features_dim = extra_features_dim
        self.encextra_size = self.encoder.hidden_size + extra_features_dim
        self.hidden = nn.ModuleList()
        for i, u in enumerate(hidden_units):
            self.hidden.append(Linear(self.encextra_size
                                      if i == 0 else hidden_units[i - 1], u))
        self.ident = Linear(self.encextra_size if len(self.hidden) == 0 else hidden_units[-1], 1)
        self.rank = nn.Sigmoid()
    def set_from_encoder(self, state_dict):
        state_dict = {k.replace('.0.', ''): v for k, v in state_dict.items()}
        self.encoder.load_state_dict(state_dict)
    def forward(self, batch):
        """n x [batch_size x (smiles|graphs) + batch_size x extra_features]"""
        res = []
        for inp in batch:       # normally 1 or 2
            if (self.extra_features_dim > 0):
                inp, extra = inp
            else:
                extra = None
            # encode molecules
            if (isinstance(inp, str)):
                inp = [mol2graph([_]) for _ in inp]
            enc = torch.cat([self.encoder(g) for g in inp], 0)  # [batch_size x 300]
            if (self.extra_features_dim > 0):
                # include extra features (e.g., chr. system)
                enc = torch.cat([enc, extra], 1)
            for h in self.hidden:
                enc = h(enc)
            res.append(self.ident(enc).transpose(0, 1)[0])      # [batch_size]
        if (self.sigmoid and len(res) == 2):
            return self.rank(res[0] - res[1]) # batch_size
        else:
            return res

def predict(x, ranker, batch_size=8192):
    ranker.eval()
    preds = []
    graphs, extra = x if len(x) == 2 else (x, None)
    with torch.no_grad():
        for i in range(np.ceil(len(graphs) / batch_size).astype(int)):
            start = i * batch_size
            end = i * batch_size + batch_size
            preds.append(ranker(((graphs[start:end],
                                  extra[start:end] if extra is not None else None), )
                                )[0].cpu().detach().numpy())
    return np.concatenate(preds)

def loss_step(ranker, x, y, weights, loss_fun):
    pred = ranker(x)
    y = torch.as_tensor(y).float().to(ranker.encoder.device)
    weights = torch.as_tensor(weights).to(ranker.encoder.device)
    if isinstance(loss_fun, nn.MarginRankingLoss):
        loss = (loss_fun(*pred, y) * weights).mean()
    else:
        loss = (loss_fun(pred, y) * weights).mean()
    return loss

def train(ranker: MPNranker, bg: BatchGenerator, epochs=2,
          writer:SummaryWriter=None, val_g: BatchGenerator=None,
          epsilon=0.5, val_writer:SummaryWriter=None,
          steps_train_loss=10, steps_val_loss=100,
          batch_size=8192, sigmoid_loss=False,
          margin_loss=0.1, early_stopping_patience=None):
    ranker.to(ranker.encoder.device)
    print('device:', ranker.encoder.device)
    optimizer = optim.Adam(ranker.parameters())
    loss_fun = (nn.BCELoss(reduction='none') if sigmoid_loss
                else nn.MarginRankingLoss(margin_loss, reduction='none'))
    ranker.train()
    loss_sum = iter_count = val_loss_sum = val_iter_count = val_pat = 0
    last_val_step = np.infty
    stop = False
    for epoch in range(epochs):
        if stop:
            break
        print(f'epoch {epoch + 1}/{epochs}')
        for x, y, weights in tqdm(bg):
            ranker.zero_grad()
            loss = loss_step(ranker, x, y, weights, loss_fun)
            loss_sum += loss.item()
            iter_count += 1
            loss.backward()
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
                        val_loss_sum += loss_step(ranker, x, y, weights, loss_fun).item()
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
        ranker.eval()
        if writer is not None:
            train_acc = eval_(bg.y, predict(bg.x, ranker, batch_size=batch_size), epsilon=epsilon)
            writer.add_scalar('acc', train_acc, iter_count)
            if (val_writer is not None):
                val_acc = eval_(val_g.y, predict(val_g.x, ranker, batch_size=batch_size), epsilon=epsilon)
                val_writer.add_scalar('acc', val_acc, iter_count)
        ranker.train()
