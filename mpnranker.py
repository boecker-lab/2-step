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


class MPNranker(nn.Module):
    def __init__(self):
        super(MPNranker, self).__init__()
        args = TrainArgs()
        args.from_dict({'dataset_type': 'classification',
                        'data_path': None})
        self.encoder = MPNEncoder(args, get_atom_fdim(), get_bond_fdim())
        self.ident = Linear(self.encoder.hidden_size, 1)
        self.rank = nn.Sigmoid()
    def set_from_encoder(self, state_dict):
        state_dict = {k.replace('.0.', ''): v for k, v in state_dict.items()}
        self.encoder.load_state_dict(state_dict)
    def forward(self, batch: Tuple[Union[List[str], List[BatchMolGraph]]]) -> torch.FloatTensor:
        res = []
        for inp in batch:       # normally 1 or 2
            if (isinstance(inp, str)):
                inp = [mol2graph([_]) for _ in batch]
            enc = torch.cat([self.encoder(g) for g in inp], 0)  # batch_size x 300
            res.append(self.ident(enc))
        return res

def predict(df, ranker, epsilon=0.5):
    ranker.eval()
    preds = ranker((df.graphs, ))
    return(eval_(df.rt, preds[0], epsilon))


def train(ranker: MPNranker, bg: BatchGenerator, epochs=2,
          writer:SummaryWriter=None, val_g: BatchGenerator=None,
          epsilon=0.5, val_writer:SummaryWriter=None):
    optimizer = optim.Adam(ranker.parameters())
    loss_fun = nn.MarginRankingLoss(0.1)
    ranker.train()
    loss_sum = iter_count = val_loss_sum = 0
    for epoch in range(epochs):
        print(f'epoch {epoch + 1}/{epochs}')
        for x, y, weights in tqdm(bg):
            ranker.zero_grad()
            pred = ranker([x[0].tolist(), x[1].tolist()])
            y[y == 0] = -1
            loss = (loss_fun(*pred, torch.Tensor([y]).transpose(0, 1)) * torch.Tensor(weights)).sum() / bg.batch_size
            loss_sum += loss.detach().item()
            iter_count += 1
            loss.backward()
            optimizer.step()
            if (iter_count % 10 == 9 and writer is not None):
                loss_avg = loss_sum / iter_count
                if writer is not None:
                    writer.add_scalar('loss', loss_avg, iter_count)
                else:
                    print(f'Loss = {loss_avg:.4f}')
            if (val_writer is not None and iter_count % 100 == 99):
                ranker.eval()
                for x, y, weights in val_g:
                    val_preds = ranker([x[0].tolist(), x[1].tolist()])
                    y[y == 0] = -1
                    val_loss_sum += (loss_fun(*val_preds, torch.Tensor([y]).transpose(0, 1))
                                     * torch.Tensor(weights)).sum().detach().item() / val_g.batch_size
                val_writer.add_scalar('loss', val_loss_sum / iter_count, iter_count)
                ranker.train()
        ranker.eval()
        if writer is not None:
            train_acc = predict(bg.get_df('graphs'), ranker, epsilon=epsilon)
            writer.add_scalar('acc', train_acc, iter_count)
            if (val_writer is not None):
                val_acc = predict(val_g.get_df('graphs'), ranker, epsilon=epsilon)
                val_writer.add_scalar('acc', val_acc, iter_count)
        ranker.train()
