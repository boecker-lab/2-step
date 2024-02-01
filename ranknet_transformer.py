import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_convert
from graphformer import graphformer
import pickle
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset
from dataclasses import dataclass, field

from chemprop.args import TrainArgs
from chemprop.models.mpn import MPN

@dataclass
class RTDataset(Dataset):
    x_mols: np.ndarray
    x_sys: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        # NOTE: for backwards compatibility also yields x_extra(which is None)
        return (self.x_mols[index], None, self.x_sys[index]), self.y[index], self.weights[index]


def dmpnn(encoder_size, depth, dropout_rate):
    args = TrainArgs()
    args.from_dict({'dataset_type': 'classification',
                    'data_path': None,
                    'hidden_size': encoder_size,
                    'depth': depth,
                    'dropout': dropout_rate,
                    'is_atom_bond_targets': True})
    model = MPN(args)
    model.name = 'dmpnn'
    return model


class RankformerGNNEmbedding(nn.Module):
    def __init__(self, ninp, nsysf, gnn_depth=3, gnn_dropout=0.0,
                 multiple_sys_tokens=False):
        super(RankformerGNNEmbedding, self).__init__()
        self.gnn = dmpnn(ninp, gnn_depth, gnn_dropout)
        self.pad_token = torch.rand((1, ninp))
        self.multiple_sys_tokens = multiple_sys_tokens
        if multiple_sys_tokens:
            self.sysf_emb = nn.Linear(1, ninp)
        else:
            self.sysf_emb = nn.Linear(nsysf, ninp)
    def forward(self, batch, encode_sys=True):
        graphs, sysf = batch
        graphs_emb = self.gnn([graphs])
        graphs_emb = [graphs_emb[0].narrow(0, a_start, a_size) for a_start, a_size in graphs_emb[1]]
        pad_len = max([x.shape[0] for x in graphs_emb])
        # NOTE: graph1 and graph2 do not share the same pad_len
        graphs_emb = torch.stack([torch.cat([g_emb] + [self.pad_token] * (pad_len - g_emb.shape[0]))
                                  for g_emb in graphs_emb])
        if encode_sys:
            if self.multiple_sys_tokens:
                sysf_emb = self.sysf_emb(torch.unsqueeze(sysf, 2))
            else:
                sysf_emb = torch.stack([self.sysf_emb(sysf)], dim=1)
        else:
            sysf_emb = None
        return (sysf_emb, graphs_emb)


class RankformerEncoder(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, nsysf, gnn_depth=3, gnn_dropout=0.0,
                 sysf_encoding_only_first_part=True, no_special_tokens=False,
                 multiple_sys_tokens=False):
        super(RankformerEncoder, self).__init__()
        self.encoder = nn.Transformer(d_model=ninp, nhead=nhead,
                                      dim_feedforward=nhid,
                                      num_encoder_layers=nlayers,
                                      num_decoder_layers=0,
                                      # might be better to not use batch_first, perhaps dmpnn output already is the not-batch_first way
                                      batch_first=True).encoder
        self.embedding = RankformerGNNEmbedding(ninp=ninp, gnn_depth=gnn_depth, gnn_dropout=gnn_dropout,
                                                nsysf=nsysf, multiple_sys_tokens=multiple_sys_tokens)
        self.no_special = no_special_tokens
        if (not no_special_tokens):
            self.cls_token = torch.rand((1, ninp))
            self.sep_token = torch.rand((1, ninp))
        self.ninp = ninp
        self.sysf_encoding_only_first_part = sysf_encoding_only_first_part
        # self.init_weights()
    def forward(self, batch, verbose=False):
        batch_size = batch[0][2].shape[0] # size of sysf of first part
        if (self.no_special):
            sentence = []
        else:
            cls_token = torch.stack([self.cls_token] * batch_size)
            sep_token = torch.stack([self.sep_token] * batch_size)
            sentence = [cls_token]
        for i, part in enumerate(batch):
            graphs, extra, sysf = part
            batch_size = sysf.shape[0]
            # ignore "extra", sysf1 must be == sysf2
            sysf_emb, graph_emb = self.embedding((graphs, sysf), encode_sys=(
                i == 0 or not self.sysf_encoding_only_first_part))
            sentence.extend(([sysf_emb] if sysf_emb is not None else []) + [graph_emb])
            # TODO: option to pad both graphs to the same size (just append pad tokens to end)
            if (not self.no_special):
                sentence.extend([sep_token])
        if verbose:
            print('sentence:', [token.shape for token in sentence])
        encoding = self.encoder(torch.cat(sentence, 1)) # (N, S, E)
        return encoding

class Rankformer(nn.Module):
    def __init__(self, ranknet_encoder, sigmoid_output=False):
        super(Rankformer, self).__init__()
        self.ranknet_encoder = ranknet_encoder
        self.ranking = nn.Linear(ranknet_encoder.ninp, 1)
        self.sigmoid_output = sigmoid_output
        self.max_epoch = 0      # track number epochs trained
        # self.init_weights()
    def forward(self, batch):
        encoding = self.ranknet_encoder(batch)
        output = self.ranking(encoding[:, 0, :]).flatten() # CLS token
        return F.sigmoid(output) if self.sigmoid_output else output

class RankformerRTPredictor(nn.Module):
    def __init__(self, ranknet_encoder, rt_hidden_dims=[64]):
        super(RankformerRTPredictor, self).__init__()
        self.ranknet_encoder = ranknet_encoder
        self.activation = F.relu
        self.hidden = nn.ModuleList()
        for i, u in enumerate(rt_hidden_dims):
            self.hidden.append(nn.Linear(ranknet_encoder.ninp if i == 0
                                      else rt_hidden_dims[i - 1], u))
        self.rt_pred = nn.Linear(rt_hidden_dims[-1], 1)
        self.max_epoch = 0      # track number epochs trained
        # self.init_weights()
    def forward(self, batch):
        encoding = self.ranknet_encoder((batch,))
        output = encoding[:, 0, :] # CLS token
        for h in self.hidden:
            output = self.activation(h(output))
        return self.rt_pred(output).flatten()


def rankformer_train(rankformer: Rankformer, bg: DataLoader, epochs=2,
                     epochs_start=0, writer:SummaryWriter=None, val_g: DataLoader=None,
                     val_writer:SummaryWriter=None,
                     confl_writer:SummaryWriter=None,
                     steps_train_loss=10, steps_val_loss=100,
                     sigmoid_loss=False,
                     early_stopping_patience=None,
                     ep_save=False, learning_rate=1e-3,
                     no_encoder_train=False):
    save_name = ('rankformer' if writer is None else
                 writer.get_logdir().split('/')[-1].replace('_train', ''))
    if (no_encoder_train):
        for p in rankformer.ranknet_encoder.parameters():
            p.requires_grad = False
    optimizer = optim.Adam(rankformer.parameters(), lr=learning_rate)
    loss_fun = nn.BCEWithLogitsLoss(reduction='none') if sigmoid_loss else nn.BCELoss(reduction='none')
    rankformer.train()
    loss_sum = iter_count = val_loss_sum = val_iter_count = val_pat = confl_loss_sum = 0
    last_val_step = np.infty
    stop = False
    for epoch in range(epochs_start, epochs_start + epochs):
        if stop:                # CTRL+C
            break
        loop = tqdm(bg)
        for x, y, weights, is_confl in loop:
            # NOTE: y has to be 0 or 1 here!
            rankformer.zero_grad()
            pred = rankformer(x)
            loss_all = loss_fun(pred, y) * weights
            loss = loss_all.mean()
            loss_sum += loss.item()
            iter_count += 1
            loss.backward()
            optimizer.step()
            if (is_confl.sum() > 0):
                confl_loss_sum += loss_all[is_confl].mean().item()
            if (iter_count % steps_train_loss == (steps_train_loss - 1) and writer is not None):
                loss_avg = loss_sum / iter_count
                if writer is not None:
                    writer.add_scalar('loss', loss_avg, iter_count)
                else:
                    print(f'Loss = {loss_avg:.4f}')
                if (confl_writer is not None):
                    confl_writer.add_scalar('loss', confl_loss_sum / iter_count, iter_count)
            if (val_writer is not None and len(val_g) > 0
                and iter_count % steps_val_loss == (steps_val_loss - 1)):
                rankformer.eval()
                with torch.no_grad():
                    for x, y, weights, is_confl in val_g:
                        pred = rankformer(x)
                        val_loss_sum += (loss_fun(pred, y) * weights).mean().item()
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
                rankformer.train()
            loop.set_description(f'Epoch [{epoch+1}/{epochs_start + epochs}]')
            loop.set_postfix(loss=loss_sum/iter_count if iter_count > 0 else np.nan,
                             val_loss=val_loss_sum/val_iter_count if val_iter_count > 0 else np.nan,
                             confl_loss=confl_loss_sum/iter_count)
        rankformer.max_epoch = epoch + 1
        if (ep_save):
            torch.save(rankformer, f'{save_name}_ep{epoch + 1}.pt')
        rankformer.train()

def rankformer_rt_train(rankformer_rt: RankformerRTPredictor, bg: DataLoader, epochs=2,
                     epochs_start=0, writer:SummaryWriter=None, val_g: DataLoader=None,
                     val_writer:SummaryWriter=None,
                     steps_train_loss=10, steps_val_loss=100,
                     early_stopping_patience=None,
                     ep_save=False, learning_rate=5e-4,
                     no_encoder_train=False):
    save_name = ('rankformer_rt' if writer is None else
                 writer.get_logdir().split('/')[-1].replace('_train', ''))
    if (no_encoder_train):
        for p in rankformer_rt.ranknet_encoder.parameters():
            p.requires_grad = False
    optimizer = optim.Adam(rankformer_rt.parameters(), lr=learning_rate)
    loss_fun = nn.L1Loss(reduction='none')
    rankformer_rt.train()
    loss_sum = iter_count = val_loss_sum = val_iter_count = val_pat = 0
    last_val_step = np.infty
    stop = False
    for epoch in range(epochs_start, epochs_start + epochs):
        if stop:                # CTRL+C
            break
        loop = tqdm(bg)
        for x, y, weights in loop:
            rankformer_rt.zero_grad()
            pred = rankformer_rt(x)
            loss_all = loss_fun(pred, y) * weights
            loss = loss_all.mean()
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
                rankformer_rt.eval()
                with torch.no_grad():
                    for x, y, weights in val_g:
                        pred = rankformer_rt(x)
                        val_loss_sum += (loss_fun(pred, y) * weights).mean().item()
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
                rankformer_rt.train()
            loop.set_description(f'Epoch [{epoch+1}/{epochs_start + epochs}]')
            loop.set_postfix(loss=loss_sum/iter_count if iter_count > 0 else np.nan,
                             val_loss=val_loss_sum/val_iter_count if val_iter_count > 0 else np.nan)
        rankformer_rt.max_epoch = epoch + 1
        if (ep_save):
            torch.save(rankformer_rt, f'{save_name}_ep{epoch + 1}.pt')
        rankformer_rt.train()

def rankformer_rt_predict(model: RankformerRTPredictor, graphs, extra, sysf, batch_size=8192,
                          prog_bar=False):
    model.eval()
    preds = []
    it = range(np.ceil(len(graphs) / batch_size).astype(int))
    if (prog_bar):
        it = tqdm(it)
    with torch.no_grad():
        for i in it:
            start = i * batch_size
            end = i * batch_size + batch_size
            graphs_batch = graphs[start:end]
            # DMPNN
            from dmpnn_graph import dmpnn_batch
            graphs_batch = dmpnn_batch(graphs_batch)
            batch = (graphs_batch, default_convert(extra[start:end]),
                     default_convert(sysf[start:end]))
            preds.append(model(batch).cpu().detach().numpy())
    return np.concatenate(preds)


if __name__ == '__main__':
    with open('/home/fleming/Documents/Projects/rtranknet/dataloader_dump.pkl', 'rb') as f:
        batch = pickle.load(f)
    ((graphs1, extra1, sysf1), (graphs2, extra2, sysf2)), y, weights, is_confl = batch
    encoder = RankformerEncoder(300, 4, 512, 2, sysf1.shape[1])
    print(f'{encoder(batch[0]).shape=}')
    encoder_multisys = RankformerEncoder(300, 4, 512, 2, sysf1.shape[1], multiple_sys_tokens=True)
    print(f'{encoder_multisys(batch[0]).shape=}')
    encoder_multisys_nospecial = RankformerEncoder(300, 4, 512, 2, sysf1.shape[1], multiple_sys_tokens=True,
                                                   no_special_tokens=True)
    print(f'{encoder_multisys_nospecial(batch[0], verbose=True).shape=}')
    encoder_nospecial = RankformerEncoder(300, 4, 512, 2, sysf1.shape[1], no_special_tokens=True)
    print(f'{encoder_nospecial(batch[0], verbose=True).shape=}')
    ranker = Rankformer(encoder)
    ranker(((graphs1, extra1, sysf1), (graphs2, extra2, sysf2)))
    rankformer_rt = RankformerRTPredictor(encoder, 16)
    rankformer_rt((graphs1, extra1, sysf1))
    # predicting
    with open('/home/fleming/Documents/Projects/rtranknet/rankformer_test_rt_rt.pt', 'rb') as f:
        rankformer_rt = torch.load(f)
    with open('/home/fleming/Documents/Projects/rtranknet/rankformer_test_rt_data.pkl', 'rb') as f:
        d = pickle.load(f)
    rankformer_rt_predict(rankformer_rt, d.train_graphs, d.train_x.astype('float32'), d.train_sys.astype('float32'))
    # other stuff
    gnn = dmpnn(300, 3, 0)
    model = RanknetTransformer(300, 4, 512, 2, sysf1.shape[1], gnn)
