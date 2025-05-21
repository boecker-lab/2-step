# from tensorboardX.writer import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import default_convert
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.modules.linear import Linear
from evaluate import eval_, eval_detailed
from utils import Data
import numpy as np
import logging
from functools import reduce
from torch.utils.data import default_collate, default_convert

logger = logging.getLogger('2-step.mpnranker2')
info = logger.info
warning = logger.warning

from utils_newbg import SPECIAL_FEATURES_SIZE

class MPNranker(nn.Module):
    def __init__(self, encoder='dmpnn', extra_features_dim=0, sys_features_dim=0,
                 hidden_units=[16, 8], hidden_units_pv=[16, 2], encoder_size=300,
                 depth=3, dropout_rate_encoder=0.0, dropout_rate_pv=0.0,
                 dropout_rate_rank=0.0, res_conn_enc=True, add_sys_features=False,
                 add_sys_features_mode=None, include_special_atom_features=False,
                 no_sys_layers=False, sys_blowup=False, no_sigmoid_roi=False):
        super(MPNranker, self).__init__()
        if (encoder == 'dmpnn'):
            from dmpnn import dmpnn
            add_dim = (sys_features_dim if add_sys_features else 0) + (SPECIAL_FEATURES_SIZE if include_special_atom_features else 0)
            self.encoder = dmpnn(encoder_size=encoder_size, depth=depth, dropout_rate=dropout_rate_encoder,
                                 add_sys_features=add_sys_features or include_special_atom_features,
                                 add_sys_features_mode=add_sys_features_mode,
                                 add_sys_features_dim=add_dim)
        else:
            raise NotImplementedError(f'{encoder} encoder')
        self.extra_features_dim = extra_features_dim
        self.sys_features_dim = sys_features_dim
        self.res_conn_enc = res_conn_enc
        self.add_sys_features = add_sys_features
        self.add_sys_features_mode = add_sys_features_mode
        self.include_special_atom_features = include_special_atom_features
        self.no_sys_layers = no_sys_layers
        self.sys_blowup = sys_blowup
        self.no_sigmoid_roi = no_sigmoid_roi
        if (not self.no_sys_layers):
            # System x molecule preference encoding
            if (self.sys_blowup):
                self.sys_blowup_layer = Linear(self.sys_features_dim, encoder_size)
                self.encextra_size = encoder_size + extra_features_dim + encoder_size
            else:
                self.encextra_size = encoder_size + extra_features_dim + sys_features_dim
            self.hidden_pv = nn.ModuleList()
            for i, u in enumerate(hidden_units_pv):
                self.hidden_pv.append(Linear(self.encextra_size if i == 0 else hidden_units_pv[i - 1], u))
            # pv dropout layer
            self.dropout_pv = nn.Dropout(dropout_rate_pv)
            last_dim = (encoder_size if res_conn_enc else 0) + extra_features_dim + hidden_units_pv[-1]
        else:
            self.encextra_size = encoder_size + extra_features_dim
            last_dim = self.encextra_size
        # actual ranking layers
        self.hidden = nn.ModuleList()
        for i, u in enumerate(hidden_units):
            self.hidden.append(Linear(last_dim if i == 0 else hidden_units[i - 1], u))
        # ranking dropout layer
        self.dropout_rank = nn.Dropout(dropout_rate_rank)
        # One ROI value for (graph,extr,sys) sample
        self.ident = Linear(hidden_units[-1], 1)
        self.max_epoch = 0      # track number epochs trained

    def forward(self, batch):
        """(1|2|n) x [batch_size x (smiles|graphs), batch_size x extra_features, batch_size x sys_features]"""
        res = []                          # TODO: no lists, just tensor stuff
        for graphs, extra, sysf in batch:       # normally 1 or 2
            if (self.encoder.name == 'dmpnn'):
                enc = self.encoder([graphs]) # [batch_size x encoder size]
            else:
                raise NotImplementedError(f'{self.encoder} encoder')
            if (not (hasattr(self, 'no_sys_layers') and self.no_sys_layers)):
                # encode system x molecule relationships
                if (hasattr(self, 'sys_blowup') and self.sys_blowup):
                    sysf = F.relu(self.sys_blowup_layer(sysf))
                enc_pv = torch.cat([enc, extra, sysf], 1)
                for h in self.hidden_pv:
                    enc_pv = F.relu(h(enc_pv))
                # apply dropout to last pv layer
                enc_pv = self.dropout_pv(enc_pv)
                # now ranking layers: [enc, enc_pv] -> ROI
                # TODO: backwards compatibility: this did not use to be an option
                if not hasattr(self, 'res_conn_enc') or self.res_conn_enc:
                    enc = torch.cat([enc, extra, enc_pv], 1)
                else:
                    enc = torch.cat([extra, enc_pv], 1)
            else:
                enc = torch.cat([extra, enc], 1)
            for h in self.hidden:
                enc = F.relu(h(enc))
            # apply dropout to last ranking layer
            enc = self.dropout_rank(enc)
            # single ROI value
            roi = self.ident(enc)
            res.append(roi.transpose(0, 1)[0])      # [batch_size]
        if (len(res) > 2):
            raise Exception('only one or two molecules are supported for now, not ', len(res))
        # return torch.sigmoid(res[0] - res[1] if len(res) == 2 else res[0])
        if (hasattr(self, 'no_sigmoid_roi') and self.no_sigmoid_roi):
            return res
        else:
            return [torch.sigmoid(r) for r in res]

    def predict(self, graphs, extra, sysf, batch_size=8192,
                prog_bar=False, ret_features=False):
        if (self.encoder.name == 'dmpnn'):
            self.eval()
        else:
            raise NotImplementedError(self.encoder)
        preds = []
        features = []
        it = range(np.ceil(len(graphs) / batch_size).astype(int))
        if (prog_bar):
            it = tqdm(it)
        with torch.no_grad():
            for i in it:
                start = i * batch_size
                end = i * batch_size + batch_size
                graphs_batch = graphs[start:end]
                if (self.encoder.name == 'dmpnn'):
                    from dmpnn_graph import dmpnn_batch
                    graphs_batch = dmpnn_batch(graphs_batch)
                else:
                    raise NotImplementedError(self.encoder)
                batch = (graphs_batch, default_convert(extra[start:end]),
                         default_convert(sysf[start:end]))
                # if (input('pdb') == 'y'):
                #     import pdb; pdb.set_trace()
                preds.append(self((batch, ))[0].cpu().detach().numpy())
                if (ret_features):
                    if (isinstance(graphs[0], str)):
                        features.extend([self.encoder([[g]]) for g in graphs[start:end]])
                    else:
                        features.extend([self.encoder([g]) for g in graphs[start:end]])
        if (ret_features):
            return np.concatenate(preds), np.concatenate(features)
        return np.concatenate(preds)
    def loss_step(self, x, y, weights, loss_fun):
        pred = self(x)
        if isinstance(loss_fun, nn.MarginRankingLoss):
            loss = ((loss_fun(*pred, y) * weights).mean(), loss_fun(*pred, y) * weights)
        else:
            loss = ((loss_fun(pred, y) * weights).mean(), loss_fun(pred, y) * weights)
        return loss


def train(ranker: MPNranker, bg: DataLoader, epochs=2,
          epochs_start=0,
          writer:SummaryWriter=None, val_g: DataLoader=None,
          epsilon=0.5, val_writer:SummaryWriter=None,
          confl_writer:SummaryWriter=None,
          steps_train_loss=10, steps_val_loss=100,
          batch_size=8192, sigmoid_loss=False,
          margin_loss=0.1, early_stopping_patience=None,
          ep_save=False, learning_rate=1e-3, adaptive_lr=False,
          gradient_clip=5, no_encoder_train=False,
          accs=True, confl_images=False, eval_train_all=True):
    if (confl_images):
        from rdkit.Chem import Draw
        from PIL import ImageDraw
    save_name = ('mpnranker' if writer is None else
                 writer.get_logdir().split('/')[-1].replace('_train', ''))
    ranker.to(ranker.encoder.device)
    print('device:', ranker.encoder.device)
    if (no_encoder_train):
        for p in ranker.encoder.parameters():
            p.requires_grad = False
    optimizer = optim.Adam(ranker.parameters(), lr=learning_rate)
    if (adaptive_lr):
        scheduler = ExponentialLR(optimizer, gamma=0.8,
                                  verbose=True)
    # loss_fun = (nn.BCELoss(reduction='none') if sigmoid_loss
    #             else nn.MarginRankingLoss(margin_loss, reduction='none'))
    # loss_fun = nn.BCEWithLogitsLoss(reduction='none')
    # loss_fun = nn.BCELoss(reduction='none')
    loss_fun = nn.MarginRankingLoss(margin_loss, reduction='none')
    if (sigmoid_loss):
        warning('sigmoid loss is not implemented anymore! margin loss will be used')
    ranker.train()
    loss_sum = iter_count = val_loss_sum = val_iter_count = val_pat = confl_loss_sum = 0
    last_val_step = np.infty
    stop = False
    val_stats, train_stats = {}, {}
    for epoch in range(epochs_start, epochs_start + epochs):
        if stop:                # CTRL+C
            break
        loop = tqdm(bg)
        for x, y, weights, is_confl in loop:
            ranker.zero_grad()
            loss = ranker.loss_step(x, y, weights, loss_fun)
            loss_sum += loss[0].item()
            iter_count += 1
            loss[0].backward()
            if (gradient_clip is not None and gradient_clip > 0):
                nn.utils.clip_grad_norm_(ranker.parameters(), gradient_clip)
            optimizer.step()
            if (is_confl.sum() > 0):
                confl_loss_sum += loss[1][is_confl].mean().item()
                high_loss = (loss[1] / weights > 2 * (loss[1] / weights)[is_confl].median().item()) & is_confl
                if (high_loss.sum() > 0 and confl_images):
                    # add pair with highest loss as images
                    high_i = np.argmax((loss[1] / weights)[high_loss].detach().numpy())
                    mols = (np.asarray(x[0][0])[high_loss][high_i].mols[0],
                            np.asarray(x[1][0])[high_loss][high_i].mols[0])
                    im1, im2 = Draw.MolToImage(mols[0]), Draw.MolToImage(mols[1])
                    ImageDraw.Draw(im1).text((20, im1.height - 20),
                                             f'loss: {(loss[1] / weights)[high_loss][high_i].item()}',
                                             fill=(0, 0, 0))
                    # import pdb; pdb.set_trace()
                    confl_writer.add_image('conflicting pair', np.concatenate(
                        [np.asarray(im1), np.asarray(im2)], 1), iter_count, dataformats='HWC')
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
                ranker.eval()
                with torch.no_grad():
                    for x, y, weights, is_confl in val_g:
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
            loop.set_description(f'Epoch [{epoch+1}/{epochs_start + epochs}]')
            loop.set_postfix(loss=loss_sum/iter_count if iter_count > 0 else np.nan,
                             val_loss=val_loss_sum/val_iter_count if val_iter_count > 0 else np.nan,
                             confl_loss=confl_loss_sum/iter_count)
        if val_writer is not None:
            val_writer.flush()
        ranker.eval()
        if accs and writer is not None:
            if (bg.dataset.dataset_info is not None):
                train_accs = []
                stats_d = {}
                for ds in set(bg.dataset.dataset_info):
                    ds_indices = [i for i, dsi in enumerate(bg.dataset.dataset_info) if dsi == ds]
                    train_acc, stats_i = eval_detailed([bg.dataset.x_ids[i] for i in ds_indices],
                        bg.dataset.y[ds_indices], ranker.predict(
                        bg.dataset.x_mols[ds_indices], bg.dataset.x_extra[ds_indices],
                            bg.dataset.x_sys[ds_indices], batch_size=batch_size), epsilon=epsilon,
                                                       void_rt=bg.dataset.void_info[ds])
                    if (not np.isnan(train_acc)):
                        train_accs.append(train_acc)
                    print(f'{ds}: \t{train_acc=:.2%}')
                    stats_d[ds] = stats_i
                train_acc = np.mean(train_accs)
                new_correct_all, new_incorrect_all, avg_roi_diff_increase_all = [], [], []
                for ds in stats_d:
                    if ds in train_stats:
                        new_correct = len({_[0] for _ in stats_d[ds]} - {_[0] for _ in train_stats[ds]})
                        new_incorrect = len({_[0] for _ in train_stats[ds]} - {_[0] for _ in stats_d[ds]})
                        roi_diff_increase = []
                        stats_d_diff, stats_diff = dict(stats_d[ds]), dict(train_stats[ds])
                        for pair, d_diff in stats_d_diff.items():
                            if pair in stats_diff:
                                roi_diff_increase.append(d_diff - stats_diff[pair])
                        avg_roi_diff_increase = np.mean(roi_diff_increase)
                        new_correct_all.append(new_correct)
                        new_incorrect_all.append(new_incorrect)
                        avg_roi_diff_increase_all.append(avg_roi_diff_increase)
                        print(f'{ds} change: +{new_correct} -{new_incorrect} ({avg_roi_diff_increase:.2f} avg. roi diff increase)')
                print(f'average total change: +{np.mean(new_correct_all):.0f} -{np.mean(new_incorrect_all):.0f}'
                      f' ({np.mean(avg_roi_diff_increase_all):.2f} avg. roi diff increase)')
                train_stats = stats_d
            else:
                train_acc = np.nan
            if (eval_train_all):
                train_acc_all = eval_(bg.dataset.y, ranker.predict(
                    bg.dataset.x_mols, bg.dataset.x_extra, bg.dataset.x_sys, batch_size=batch_size), epsilon=epsilon)
                writer.add_scalar('acc_all', train_acc_all, iter_count)
            else:
                train_acc_all = np.nan
            writer.add_scalar('acc', train_acc, iter_count)
            writer.flush()
            print(f'{train_acc=:.2%}, {train_acc_all=:.2%}')
            if (val_writer is not None):
                if (val_g.dataset.dataset_info is not None):
                    val_accs = []
                    stats_d = {}
                    for ds in set(val_g.dataset.dataset_info):
                        ds_indices = [i for i, dsi in enumerate(val_g.dataset.dataset_info) if dsi == ds]
                        val_acc, stats_i = eval_detailed([val_g.dataset.x_ids[i] for i in ds_indices],
                            val_g.dataset.y[ds_indices], ranker.predict(
                            val_g.dataset.x_mols[ds_indices], val_g.dataset.x_extra[ds_indices],
                                val_g.dataset.x_sys[ds_indices], batch_size=batch_size), epsilon=epsilon,
                                                         void_rt=val_g.dataset.void_info[ds])
                        if (not np.isnan(val_acc)):
                            val_accs.append(val_acc)
                        print(f'{ds}: \t{val_acc=:.2%}')
                        stats_d[ds] = stats_i
                    val_acc = np.mean(val_accs)
                    new_correct_all, new_incorrect_all, avg_roi_diff_increase_all = [], [], []
                    for ds in stats_d:
                        if ds in val_stats:
                            new_correct = len({_[0] for _ in stats_d[ds]} - {_[0] for _ in val_stats[ds]})
                            new_incorrect = len({_[0] for _ in val_stats[ds]} - {_[0] for _ in stats_d[ds]})
                            roi_diff_increase = []
                            stats_d_diff, stats_diff = dict(stats_d[ds]), dict(val_stats[ds])
                            for pair, d_diff in stats_d_diff.items():
                                if pair in stats_diff:
                                    roi_diff_increase.append(d_diff - stats_diff[pair])
                            avg_roi_diff_increase = np.mean(roi_diff_increase)
                            new_correct_all.append(new_correct)
                            new_incorrect_all.append(new_incorrect)
                            avg_roi_diff_increase_all.append(avg_roi_diff_increase)
                            print(f'{ds} change: +{new_correct} -{new_incorrect} ({avg_roi_diff_increase:.2f} avg. roi diff increase)')
                    print(f'average total change: +{np.mean(new_correct_all):.0f} -{np.mean(new_incorrect_all):.0f}'
                          f' ({np.mean(avg_roi_diff_increase_all):.2f} avg. roi diff increase)')
                    val_stats = stats_d
                else:
                    val_acc = eval_(val_g.dataset.y, ranker.predict(val_g.dataset.x_mols, val_g.dataset.x_extra, val_g.dataset.x_sys,
                                                            batch_size=batch_size), epsilon=epsilon)
                val_writer.add_scalar('acc', val_acc, iter_count)
                val_writer.flush()
                print(f'{val_acc=:.2%}')
        ranker.max_epoch = epoch + 1
        if (ep_save):
            torch.save(ranker, f'{save_name}_ep{epoch + 1}.pt')
        if (adaptive_lr):
            scheduler.step()
        ranker.train()

def custom_collate(batch):
    return (                    # x, y, weights, is_confl
                (
                    (custom_collate.graph_batch([_[0][0][0] for _ in batch]),
                     torch.stack(list(map(default_convert, [_[0][0][1] for _ in batch])), 0),
                     torch.stack(list(map(default_convert, [_[0][0][2] for _ in batch])), 0)),
                    (custom_collate.graph_batch([_[0][1][0] for _ in batch]),
                     torch.stack(list(map(default_convert, [_[0][1][1] for _ in batch])), 0),
                     torch.stack(list(map(default_convert, [_[0][1][2] for _ in batch])), 0))
                ),
                torch.stack(list(map(default_convert, [_[1] for _ in batch])), 0),
                torch.stack(list(map(default_convert, [_[2] for _ in batch])), 0),
                torch.stack(list(map(default_convert, [_[3] for _ in batch])), 0))

def custom_collate_single(batch):
    transformed = ((custom_collate_single.graph_batch([_[0][0] for _ in batch]),
                    None,
                    torch.stack(list(map(default_convert, [_[0][2] for _ in batch])), 0)),
                   torch.stack(list(map(default_convert, [_[1] for _ in batch])), 0),
                   torch.stack(list(map(default_convert, [_[2] for _ in batch])), 0))
    return transformed
