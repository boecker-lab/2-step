import math
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


def dmpnn(encoder_size, depth, dropout_rate, no_reduce=True):
    args = TrainArgs()
    args.from_dict({'dataset_type': 'classification',
                    'data_path': None,
                    'hidden_size': encoder_size,
                    'depth': depth,
                    'dropout': dropout_rate,
                    'is_atom_bond_targets': no_reduce})
    model = MPN(args)
    model.name = 'dmpnn'
    return model


class RankformerGNNEmbedding(nn.Module):
    def __init__(self, ninp, nsysf, gnn_depth=3, gnn_dropout=0.0,
                 multiple_sys_tokens=False, one_token_per_graph=False):
        super(RankformerGNNEmbedding, self).__init__()
        self.one_token_per_graph = one_token_per_graph
        self.gnn = dmpnn(ninp, gnn_depth, gnn_dropout,
                         no_reduce=not one_token_per_graph)
        self.pad_token = nn.Parameter(torch.randn(1, ninp))
        self.multiple_sys_tokens = multiple_sys_tokens
        if multiple_sys_tokens:
            self.sysf_emb = nn.Linear(1, ninp)
        else:
            self.sysf_emb = nn.Linear(nsysf, ninp)
    def forward(self, batch, encode_sys=True):
        graphs, sysf = batch
        graphs_emb = self.gnn([graphs])
        if self.one_token_per_graph:
            graphs_emb = graphs_emb.unsqueeze(1)
        else:
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
    def __init__(self, ninp, nhead, nhid, nlayers, nsysf, dropout=0.1, gnn_depth=3, gnn_dropout=0.0,
                 sysf_encoding_only_first_part=True, no_special_tokens=False,
                 multiple_sys_tokens=False, one_token_per_graph=False, mol_order_embedding=True):
        super(RankformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=ninp, nhead=nhead, dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True # might be better to not use batch_first, perhaps dmpnn output already is the not-batch_first way
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = RankformerGNNEmbedding(ninp=ninp, gnn_depth=gnn_depth, gnn_dropout=gnn_dropout,
                                                nsysf=nsysf, multiple_sys_tokens=multiple_sys_tokens,
                                                one_token_per_graph=one_token_per_graph)
        self.mol_order_embedding = mol_order_embedding
        if mol_order_embedding:
            self.mol_order = nn.Embedding(2, 1)
        self.no_special = no_special_tokens
        if (not no_special_tokens):
            self.cls_token = nn.Parameter(torch.randn(1, ninp))
            self.sep_token = nn.Parameter(torch.randn(1, ninp))
        self.ninp = ninp
        self.sysf_encoding_only_first_part = sysf_encoding_only_first_part
        # self.init_weights()
    def forward(self, batch, verbose=False):
        batch_size = batch[0][2].shape[0] # size of sysf of first part
        if (self.no_special):
            sentence = []
            sentence_order = []
        else:
            cls_token = torch.stack([self.cls_token] * batch_size)
            sep_token = torch.stack([self.sep_token] * batch_size)
            sentence = [cls_token]
            sentence_order = [0] * sum([x.shape[1] for x in sentence])
        for i, part in enumerate(batch):
            graphs, extra, sysf = part
            batch_size = sysf.shape[0]
            # ignore "extra", sysf1 must be == sysf2
            sysf_emb, graph_emb = self.embedding((graphs, sysf), encode_sys=(
                i == 0 or not self.sysf_encoding_only_first_part))
            extension = ([sysf_emb] if sysf_emb is not None else []) + [graph_emb]
            # TODO: option to pad both graphs to the same size (just append pad tokens to end)
            if (not self.no_special):
                extension += [sep_token]
            sentence.extend(extension)
            sentence_order += [i] * sum([x.shape[1] for x in extension])
        if verbose:
            print('sentence:', [token.shape for token in sentence])
            print('mol order:', sentence_order)
        encoding = torch.cat(sentence, 1)
        if self.mol_order_embedding:
            mol_order_embedding = self.mol_order(torch.tensor(sentence_order, dtype=torch.int))
            encoding += mol_order_embedding
        encoding *= math.sqrt(self.ninp)
        encoding = self.encoder(encoding) # (N, S, E)
        return encoding

class FFNEncoder(nn.Module):
    def __init__(self, ninp, nsysf, gnn_depth=3, gnn_dropout=0.0,
                 no_special_tokens=False):
        super(FFNEncoder, self).__init__()
        self.no_special = no_special_tokens
        if no_special_tokens:
            sentence_len = 3        # SYS, G1, G2
        else:
            sentence_len = 6        # CLS, SYS, G1, SEP, G2, SEP
        self.encoder = nn.Linear(ninp * sentence_len, ninp)
        self.embedding = RankformerGNNEmbedding(ninp=ninp, gnn_depth=gnn_depth, gnn_dropout=gnn_dropout,
                                                nsysf=nsysf, multiple_sys_tokens=False,
                                                one_token_per_graph=True)
        self.cls_token = nn.Parameter(torch.randn(1, ninp))
        self.sep_token = nn.Parameter(torch.randn(1, ninp))
        self.ninp = ninp
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
            sysf_emb, graph_emb = self.embedding((graphs, sysf), encode_sys=(i == 0))
            sentence.extend(([sysf_emb] if sysf_emb is not None else []) + [graph_emb])
            if (not self.no_special):
                sentence.extend([sep_token])
        if verbose:
            print('sentence:', [token.shape for token in sentence])
        encoding = self.encoder(torch.cat(sentence, 1).flatten(1)).unsqueeze(1) # (N, 1, E)
        return encoding

class Rankformer(nn.Module):
    def __init__(self, ranknet_encoder, sigmoid_output=False, hidden_dims=[]):
        super(Rankformer, self).__init__()
        self.ranknet_encoder = ranknet_encoder
        if len(hidden_dims) > 0:
            self.activation = F.relu
            self.hidden = nn.ModuleList()
            for i, u in enumerate(hidden_dims):
                self.hidden.append(nn.Linear(ranknet_encoder.ninp if i == 0
                                             else hidden_dims[i - 1], u))
        self.ranking = nn.Linear(ranknet_encoder.ninp if len(hidden_dims) == 0
                                 else hidden_dims[-1], 1)
        self.sigmoid_output = sigmoid_output
        self.max_epoch = 0      # track number epochs trained
        # self.init_weights()
    def forward(self, batch):
        output = self.ranknet_encoder(batch)[:, 0, :]
        if (hasattr(self, 'hidden')):
            for h in self.hidden:
                output = self.activation(h(output))
        output = self.ranking(output).flatten() # CLS token
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
                     no_weights=False,
                     early_stopping_patience=None,
                     ep_save=False, learning_rate=1e-3,
                     no_encoder_train=False, calc_acc=True,
                     clip_gradient=1.):
    save_name = ('rankformer' if writer is None else
                 writer.get_logdir().split('/')[-1].replace('_train', ''))
    if (no_encoder_train):
        for p in rankformer.ranknet_encoder.parameters():
            p.requires_grad = False
    # optimizer = optim.Adam(rankformer.parameters(), lr=learning_rate)
    optimizer = optim.RAdam(rankformer.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(rankformer.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    loss_kwargs = dict(reduction='none')
    loss_fun = nn.BCEWithLogitsLoss(**loss_kwargs) if sigmoid_loss else nn.BCELoss(**loss_kwargs)
    # loss_fun = nn.L1Loss(reduction='none')
    rankformer.train()
    loss_sum = iter_count = val_loss_sum = val_iter_count = val_pat = confl_loss_sum = 0
    train_acc = val_acc = confl_acc = np.nan
    train_acc_sum = val_acc_sum = confl_acc_sum = 0
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
            if not (((pred >= 0) & (pred <= 1)).all().item()):
                print(pred)
            if not (((y >= 0) & (y <= 1)).all().item()):
                print(y)
            # print(pred)
            # print('\nx1', [g.smiles for g in x[0][0].mol_graphs])
            # print('x2', [g.smiles for g in x[1][0].mol_graphs])
            # print('y', y)
            # DEBUG: change y so it is simply which mol has more atoms
            # y = torch.from_numpy(np.array([g1.n_atoms > g2.n_atoms for g1, g2 in zip(x[0][0].mol_graphs, x[1][0].mol_graphs)], dtype='float32'))
            loss_all = loss_fun(pred, y)
            if not (no_weights):
                loss_all *= weights
            if (calc_acc):
                train_acc = (torch.isclose(pred, y, atol=0.499999).sum() / len(y)).item()
                train_acc_sum += train_acc
                if writer is not None:
                    writer.add_scalar('acc', train_acc_sum/iter_count
                                      if iter_count > 0 else np.nan, iter_count)
            loss = loss_all.mean()
            loss_sum += loss.item()
            iter_count += 1
            loss.backward()
            if (clip_gradient is not None):
                nn.utils.clip_grad_norm_(rankformer.parameters(), clip_gradient)
            optimizer.step()
            if (is_confl.sum() > 0):
                confl_loss_sum += loss_all[is_confl].mean().item()
                if (calc_acc):
                    confl_acc = (torch.isclose(pred[is_confl], y[is_confl], atol=0.499999).sum()
                                 / is_confl.sum()).item()
                    confl_acc_sum += confl_acc
                    if confl_writer is not None:
                        confl_writer.add_scalar('acc', confl_acc_sum/iter_count
                                                if iter_count > 0 else np.nan, iter_count)
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
                        val_loss_all = loss_fun(pred, y)
                        if not (no_weights):
                            val_loss_all *= weights
                        val_loss_sum += val_loss_all.mean().item()
                        if (calc_acc):
                            val_acc = (torch.isclose(pred, y, atol=0.499999).sum() / len(y)).item()
                            val_acc_sum += val_acc
                            if val_writer is not None:
                                val_writer.add_scalar('acc', val_acc_sum/val_iter_count
                                                      if val_iter_count > 0 else np.nan, iter_count)
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
                             confl_loss=confl_loss_sum/iter_count,
                             **(dict(train_acc=train_acc_sum/iter_count if iter_count > 0 else np.nan,
                                     val_acc=val_acc_sum/val_iter_count if val_iter_count > 0 else np.nan,
                                     confl_acc=confl_acc_sum/iter_count if iter_count > 0 else np.nan)
                                if calc_acc else dict()))
        # scheduler.step()
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
            # loss_all = loss_fun(pred, y / 100) * weights # TODO: maybe needed
            # or other way around: y times 100
            # loss_all = loss_fun(pred, y * 100) * weights # TODO: maybe needed
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

def rankformer_eval(rankformer, bg, max_batches_count=None):
    rankformer.eval()
    correct = []
    with torch.no_grad():
        loop = tqdm(enumerate(bg), total=max_batches_count)
        for i, (x, y, weights, is_confl) in loop:
            if max_batches_count and i >= max_batches_count:
                break
            try:
                pred = rankformer(x)
                correct.extend(torch.isclose(pred, y, atol=0.499999).detach().tolist())
            except KeyboardInterrupt:
                break
            # val_loss_sum += (loss_fun(pred, y) * weights).mean().item()
    return sum(correct) / len(correct)

def test_rankformer_rt():
    rankformer_model = torch.load('/home/fleming/Documents/Projects/rtranknet/runs/rankformer_small_ep2.pt', map_location=torch.device('cpu'))
    rankformer_encoder = rankformer_model.ranknet_encoder
    rankformer_rt = RankformerRTPredictor(rankformer_encoder, [64])
    with open('/home/fleming/Documents/Projects/rtranknet/runs/rankformer_small_data.pkl', 'rb') as f:
        data = pickle.load(f)
    from train import prepare_rt_data
    from mpnranker2 import custom_collate_single as custom_collate
    from dmpnn_graph import dmpnn_batch
    custom_collate.graph_batch = dmpnn_batch
    train_ds, val_ds = prepare_rt_data(data, data.train_graphs, data.train_sys, data.train_y,
                                       data.val_graphs, data.val_sys, data.val_y)
    trainloader = DataLoader(train_ds, 1024, shuffle=True, collate_fn=custom_collate)
    (graphs, extra, sysf), y, weights = next(iter(trainloader))
    rankformer_rt((graphs, extra, sysf))
    # trained rankformer_rt
    rankformer_rt = torch.load('/home/fleming/Documents/Projects/rtranknet/rankformer_small_rt_ep5.pt')
    preds = rankformer_rt((graphs, extra, sysf))
    print(f'{(preds - y).abs().mean()=}')
    # plot, but only for one dataset
    train_df = data.df.iloc[data.train_indices].set_index(data.train_indices)
    ds = train_df.dataset_id.iloc[0]
    ds_select = (train_df.dataset_id == '0239')
    ds_graphs, ds_sys, ds_y = data.train_graphs[ds_select], data.train_sys[ds_select], data.train_y[ds_select]
    ds_ds = RTDataset(ds_graphs, ds_sys.astype('float32'), ds_y.astype('float32'), np.ones_like(ds_y).astype('float32'))
    ds_loader = DataLoader(ds_ds, 1024, shuffle=False, collate_fn=custom_collate)
    (graphs, extra, sysf), y, weights = next(iter(ds_loader))
    preds = rankformer_rt((graphs, extra, sysf))
    print(f'{(preds / 100 - y).abs().mean()=}')
    rankformer_rt_train(rankformer_rt, trainloader, 1, 0,  no_encoder_train=True)

def load_rankformer(path, device):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.ranknet_encoder.embedding.gnn.device = device
    model.ranknet_encoder.embedding.gnn.encoder[0].device = device
    return model

def test_rankformer():
    from utils_newbg import RankDataset, check_integrity
    from mpnranker2 import custom_collate
    from dmpnn_graph import dmpnn_batch
    custom_collate.graph_batch = dmpnn_batch
    with open('/home/fleming/Documents/Projects/rtranknet/runs/rankformer_data.pkl', 'rb') as f:
        data = pickle.load(f)
    traindata = RankDataset(x_mols=data.train_graphs, x_extra=data.train_x, x_sys=data.train_sys,
                            x_ids=data.df.iloc[data.train_indices].smiles.tolist(),
                            y=data.train_y, x_sys_global_num=data.x_info_global_num,
                            dataset_info=data.df.dataset_id.iloc[data.train_indices].tolist(),
                            void_info=data.void_info, cluster=True, y_neg=False, y_float=True,
                            max_indices_size=10_000)
    # with open('/home/fleming/Documents/Projects/rtranknet/runs/rankformer_dataset.pkl', 'wb') as out:
    #     pickle.dump(traindata, out)
    # _, to_clean, _ = check_integrity(traindata, clean=True)
    # traindata.remove_indices(to_clean)
    trainloader = DataLoader(traindata, 128, shuffle=True, collate_fn=custom_collate)
    rankformer_model = load_rankformer('/home/fleming/Documents/Projects/rtranknet/runs/rankformer.pt', torch.device('cpu'))
    rankformer_model = load_rankformer('/home/fleming/Documents/Projects/rtranknet/runs/rankformer_nospecial.pt', torch.device('cpu'))
    rankformer_model = load_rankformer('/home/fleming/Documents/Projects/rtranknet/runs/rankformer_multisys.pt', torch.device('cpu'))
    print(f'{rankformer_eval(rankformer_model, trainloader, max_batches_count=20)=}')

def construct_sample(g1, g2, y, sysf=0):
    arr = lambda x: np.array(x, dtype='float32')
    return (((g1, arr([-1]), arr([sysf])),
             (g2, arr([-1]), arr([sysf]))),
            arr([y]), arr([1]), arr([1]))

def check_order(ds, s1, s2):
    return cs_df.loc[(cs_df.dataset_id == ds)].set_index('smiles.std').loc[[s1, s2], 'rt']

def test_embedding():
    smiles_list = [
        'CC',
        'CO',
        'CN',
        'COC',
        'C=C'
    ]
    from dmpnn_graph import dmpnn_batch, dmpnn_graph
    graphs = [dmpnn_graph(s) for s in smiles_list]
    sample1 = [
        construct_sample(graphs[3], graphs[2], 1),
        construct_sample(graphs[0], graphs[1], 0),
    ]
    # sample1 = ((graphs[0:2], None, np.array([0])), (graphs[2:4], None, np.array([0]))), np.array([0, 1]), np.array([1, 1])
    from mpnranker2 import custom_collate
    custom_collate.graph_batch = dmpnn_batch
    sample1_batch = custom_collate(sample1)
    encoder = RankformerEncoder(6, 2, 2, 1, 1, 1, one_token_per_graph=True)
    fake_encoder = FFNEncoder(6, 1)
    rankformer = Rankformer(encoder)
    fake_rankformer = Rankformer(fake_encoder)
    # how is the first batch embedded?
    semb, gemb = encoder.embedding((sample1_batch[0][0][0], sample1_batch[0][0][2]))
    gemb_gnn = encoder.embedding.gnn(([sample1_batch[0][0][0]]))
    # whole sample
    encoder(sample1_batch[0], verbose=True)



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
