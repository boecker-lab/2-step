from utils_newbg import RankDataset
import pickle
from train import preprocess, TrainArgs
from utils import Data
from torch.utils.data import DataLoader
from utils_datasetpairs import DatasetPairs
from twinpairs import TwinPairs, TwinPair
import numpy as np
import pandas as pd
from mpnranker2 import MPNranker, RankerTwins, data_eval
import torch

def ex_data():
    data = Data(use_system_information=True,
                metadata_void_rt=True,
                custom_features=['MolLogP'],
                use_hsm=True,
                graph_mode=True)
    for did in ['0068', '0138']:
        data.add_dataset_id(did, isomeric=True)
    return data

def test_newbg():
    data = ex_data()

    ((train_graphs, train_x, train_sys, train_y),
     (val_graphs, val_x, val_sys, val_y),
     (test_graphs, test_x, test_sys, test_y)) = preprocess(data, TrainArgs().parse_args(
         '--input 0001'.split()))

    traindata = RankDataset(x_mols=train_graphs, x_extra=train_x, x_sys=train_sys,
                            x_ids=data.df.iloc[data.train_indices].smiles.tolist(),
                            y=train_y, dataset_info=data.df.dataset_id.iloc[data.train_indices].tolist(),
                            void_info=data.void_info)
    trainloader = DataLoader(traindata, 32, shuffle=True)
    return trainloader


def find_confl_preds():
    confl_a = {}
    confl_b = {}
    for i in range(len(td.x1_indices)):
        p = frozenset([td.x_ids[td.x1_indices[i]], td.x_ids[td.x2_indices[i]]])
        if (p in pairs):
            if (td.dataset_info[td.x1_indices[i]] == '0068'):
                confl_a[p] = i
            else:
                confl_b[p] = i
    confl = {k: (confl_a[k], confl_b[k]) for k in set(confl_a) & set(confl_b)}

def test_twins(for_other_test=False):
    data = ex_data()

    _ = preprocess(data, TrainArgs().parse_args('--input 0001 --test_split 0 --val_split 0'.split()))

    ds1_i = [data.df.index.get_loc(i) for i in data.df.loc[data.df.dataset_id == '0068'].index]
    ds2_i = [data.df.index.get_loc(i) for i in data.df.loc[data.df.dataset_id == '0138'].index]

    confl_pairs = pickle.load(open('/home/fleming/Documents/Uni/RTpred/pairs_0068_0138.pkl', 'rb'))

    # first test whether all those pairs really are included in both dps
    if (not for_other_test):
        i = 0
        for compounds in confl_pairs:
            if (len(compounds) != 2):
                print(compounds)
                continue
            c1, c2 = compounds
            i += 1
            try:
                print('\t'.join([str(i), c1, c2,
                                 str(data.df.loc[(data.df.dataset_id == '0068') &
                                                 (data.df['smiles.std'] == c1), 'rt'].iloc[0]),
                                 str(data.df.loc[(data.df.dataset_id == '0068') &
                                                 (data.df['smiles.std'] == c2), 'rt'].iloc[0]),
                                 str(data.df.loc[(data.df.dataset_id == '0138') &
                                                 (data.df['smiles.std'] == c1), 'rt'].iloc[0]),
                                 str(data.df.loc[(data.df.dataset_id == '0138') &
                                                 (data.df['smiles.std'] == c2), 'rt'].iloc[0])
                                 ]))
            except:
                print('\t'.join([str(i), c1, c2]))
            print((c1 in dp1.x_ids and c2 in dp1.x_ids))

    dp1 = DatasetPairs(x_mols=data.df['smiles.std'].iloc[ds1_i].tolist(),
                       x_extra=data.x_features[ds1_i],
                       x_sys=data.x_info[ds1_i[0]],
                       x_ids=data.df['smiles.std'].iloc[ds1_i].tolist(),
                       y=data.get_y()[ds1_i], dataset_id='0068',
                       void=data.df['column.t0'].iloc[ds1_i[0]],
                       conflicting_smiles_pairs=confl_pairs)
    dp2 = DatasetPairs(x_mols=data.df['smiles.std'].iloc[ds2_i].tolist(),
                       x_extra=data.x_features[ds2_i],
                       x_sys=data.x_info[ds2_i[0]],
                       x_ids=data.df['smiles.std'].iloc[ds2_i].tolist(),
                       y=data.get_y()[ds2_i], dataset_id='0138',
                       void=data.df['column.t0'].iloc[ds2_i[0]],
                       conflicting_smiles_pairs=confl_pairs)

    from rdkit.Chem.Descriptors import MolLogP
    from rdkit.Chem import MolFromSmiles

    assert np.isclose(np.asarray([[MolLogP(MolFromSmiles(s))] for s in dp1.x_ids]),
                      dp1.x_extra).all(), 'extra/smiles order wrong'

    tps = TwinPairs({'0068': dp1, '0138': dp2}, confl_pairs)

    if (for_other_test):
        return tps, data

    # now, sample some twin pairs, ensure that the 50/25/25 rule is somewhat correct
    twins = []
    from random import sample
    for i in sample(range(len(tps)), 300):
        tp = tps[i]
        type_ = twin_type(tp)
        twins.append({'i': i, 'confl': type_[0] == 'confl', 'nonconfl_type': type_[1],
                      'y': tp.confl, 'y1': tp.p1.y, 'y2': tp.p2.y})
    sampled = pd.DataFrame.from_records(twins, index='i').astype('category')
    for c in sampled.columns:
        print(sampled[c].value_counts() / 3)
    # -> looking very good! 🤗

def twin_type(tp: TwinPair):
    if (tp.confl == 1):
        return ('confl', None)
    if (len(set([tp.p1.x1.x, tp.p1.x2.x, tp.p2.x1.x, tp.p2.x2.x])) == 2):
        return ('nonconfl', 'same')
    return ('nonconfl', 'diff')

def test_rankertwins():
    tps, data = test_twins(True)
    loader = DataLoader(tps, 32, True)
    twins = RankerTwins(extra_features_dim=1, sys_features_dim=9,
                        hidden_units=[32, 16, 8])

    from mpnranker2 import twin_train
    twin_train(twins, 20, train_loader=loader, confl_mod=0.6)
    torch.save(twins.ranker, 'twinranker1.pt')

    # visualization
    # from torchviz import make_dot
    # make_dot((batch.p1.y, batch.p2.y), params=dict(list(twins.named_parameters()))).render(
    #     "twinranker_torchviz", format="png")
    # torch.onnx.export(twins, x, 'twinranker.onnx')

    # testing batches etc.
    batch = list(loader)[0]
    x = ((batch.p1.x1, batch.p1.x2), (batch.p2.x1, batch.p2.x2))
    twins(x)
    rank_loss = torch.nn.BCELoss(reduction='none')
    twin_loss = torch.nn.BCELoss(reduction='none')
    loss = twins.loss_step(x[0], x[1], batch.p1.y, batch.p2.y,
                           batch.p1.weights, batch.p2.weights,
                           batch.confl, batch.weights,
                           rank_loss, twin_loss)

    # prediction
    from mpnranker2 import data_eval
    print(data_eval(twins.ranker, data))


def toy_confl_example():
    # build and train twinranker
    tps, data = test_twins(True)
    loader = DataLoader(tps, 32, True)
    twins = RankerTwins(extra_features_dim=1, sys_features_dim=9,
                        hidden_units=[32, 16, 8])

    from mpnranker2 import twin_train
    twin_train(twins, 10, train_loader=loader, confl_mod=0.6)
    torch.save(twins.ranker, 'twinranker1.pt')


    batch = list(loader)[0]
    x = ((batch.p1.x1, batch.p1.x2), (batch.p2.x1, batch.p2.x2))
    twins(x)
    rank_loss = torch.nn.BCELoss(reduction='none')
    twin_loss = torch.nn.BCELoss(reduction='none')
    loss = twins.loss_step(x[0], x[1], batch.p1.y, batch.p2.y,
                           batch.p1.weights, batch.p2.weights,
                           batch.confl, batch.weights,
                           rank_loss, twin_loss)

    # example
    # confl sample: batch[2] == 1
    # confl sample: batch[2] == 0
    i_confl = int(batch[2].argmax())
    i_nonconfl = int(batch[2].argmin())
    x_confl = ((((batch.p1.x1.x[i_confl],), batch.p1.x1.extra[[i_confl]], batch.p1.x1.sys[[i_confl]]),
                ((batch.p1.x2.x[i_confl],), batch.p1.x2.extra[[i_confl]], batch.p1.x2.sys[[i_confl]])),
               (((batch.p2.x1.x[i_confl],), batch.p2.x1.extra[[i_confl]], batch.p2.x1.sys[[i_confl]]),
                ((batch.p2.x2.x[i_confl],), batch.p2.x2.extra[[i_confl]], batch.p2.x2.sys[[i_confl]])))
    x_confl_loss = twins.loss_step(x_confl[0], x_confl[1], batch.p1.y[[i_confl]], batch.p2.y[[i_confl]],
                                   [1], [1], batch.confl[[i_confl]], [1],
                                   rank_loss, twin_loss)
    (x_confl_loss[0] + x_confl_loss[1]).backward()
    # the whole gradient related to system weights
    x_confl_sysgrad = (list(twins.ranker.hidden.named_parameters())[0][1].grad[:, -9:] ** 2).sum()

    # nonconfl example
    x_nonconfl = ((((batch.p1.x1.x[i_nonconfl],), batch.p1.x1.extra[[i_nonconfl]], batch.p1.x1.sys[[i_nonconfl]]),
                ((batch.p1.x2.x[i_nonconfl],), batch.p1.x2.extra[[i_nonconfl]], batch.p1.x2.sys[[i_nonconfl]])),
               (((batch.p2.x1.x[i_nonconfl],), batch.p2.x1.extra[[i_nonconfl]], batch.p2.x1.sys[[i_nonconfl]]),
                ((batch.p2.x2.x[i_nonconfl],), batch.p2.x2.extra[[i_nonconfl]], batch.p2.x2.sys[[i_nonconfl]])))
    x_nonconfl_loss = twins.loss_step(x_nonconfl[0], x_nonconfl[1], batch.p1.y[[i_nonconfl]], batch.p2.y[[i_nonconfl]],
                                   [1], [1], batch.confl[[i_nonconfl]], [1],
                                   rank_loss, twin_loss)
    (x_nonconfl_loss[0] + x_nonconfl_loss[1]).backward()
    # the whole gradient related to system weights
    x_nonconfl_sysgrad = (list(twins.ranker.hidden.named_parameters())[0][1].grad[:, -9:] ** 2).sum()

    sysgrad_for_batchindex(twins, batch, 0)

    for i in range(len(batch[2])):
        (y1, y2, yconfl), (r1_sys, r2_sys, c_sys) = sysgrad_for_batchindex(twins, batch, i)
        print(f'pair 1: |{int(y1[1])} - {float(y1[0]):.2f}| = {np.abs(y1[1]-y1[0])[0]:.2f} → {r1_sys:.2f}')
        print(f'pair 2: |{int(y2[1])} - {float(y2[0]):.2f}| = {np.abs(y2[1]-y2[0])[0]:.2f} → {r2_sys:.2f}')
        print(f'confli: |{int(yconfl[1])} - {float(yconfl[0]):.2f}| = {np.abs(yconfl[1]-yconfl[0])[0]:.2f} → {c_sys:.2f}\n')


def sysgrad_for_batchindex(twins, batch, index):
    rank_loss = torch.nn.BCELoss(reduction='none')
    twin_loss = torch.nn.BCELoss(reduction='none')
    x = ((((batch.p1.x1.x[index],), batch.p1.x1.extra[[index]], batch.p1.x1.sys[[index]]),
                ((batch.p1.x2.x[index],), batch.p1.x2.extra[[index]], batch.p1.x2.sys[[index]])),
               (((batch.p2.x1.x[index],), batch.p2.x1.extra[[index]], batch.p2.x1.sys[[index]]),
                ((batch.p2.x2.x[index],), batch.p2.x2.extra[[index]], batch.p2.x2.sys[[index]])))
    # all losses
    y1_pred, y2_pred, yconfl_pred = twins((x[0], x[1]))
    y1, y2, yconfl = [torch.as_tensor(_).float().to(twins.ranker.encoder.device)
                      for _ in [batch.p1.y[[index]], batch.p2.y[[index]], batch.confl[[index]]]]
    ranker1_loss = rank_loss(y1_pred, y1)
    ranker2_loss = rank_loss(y2_pred, y2)
    twins_confl_loss = twin_loss(yconfl_pred, yconfl)
    ret = []
    for loss in (ranker1_loss, ranker2_loss, twins_confl_loss):
        twins.zero_grad()
        loss.backward(retain_graph=True)
        ret.append((list(twins.ranker.hidden.named_parameters())[0][1].grad[:, -9:] ** 2).sum())
    return ((y1_pred.detach(), y1.detach()), (y2_pred.detach(), y2.detach()), (yconfl_pred.detach(), yconfl.detach())), ret

def test_new_ranker():
    ranker = MPNranker(1, 9, [16, 8])
    tps, data = test_twins(True)
    loader = DataLoader(tps, 32, True)
    batch = list(loader)[0]
    x = (batch.p1.x1, batch.p1.x2)
    ranker.loss_step
    from mpnranker2 import train as ranker_train
    tlll = test_newbg()
    ranker_train(ranker, tlll, epochs=5)
    batch = list(tlll)[1]
    x, y, weights = batch
    ranker((x[0], x[2]))
    ranker((x[1], x[2]))
    (ranker((x[0], x[2])) < ranker((x[1], x[2])))
    ranker(x) < 0.5
    (ranker((x[0], x[2])) < ranker((x[1], x[2]))) == (ranker(x) < 0.5)

def test_new_ranker_single():
    root_folder = '/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/'
    # data
    data = Data(use_system_information=True,
                metadata_void_rt=True,
                custom_features=['MolLogP'],
                use_hsm=True,
                custom_column_fields='column.flowrate column.length column.id'.split(),
                graph_mode=True,
                smiles_for_graphs=True,
                repo_root_folder=root_folder)
    dss = ['0001']
    for did in dss:
        data.add_dataset_id(did, isomeric=True, repo_root_folder=root_folder)
    ((train_graphs, train_x, train_sys, train_y),
     (val_graphs, val_x, val_sys, val_y),
     (test_graphs, test_x, test_sys, test_y)) = preprocess(data, TrainArgs().parse_args(
         '--input 0001'.split()))
    traindata = RankDataset(x_mols=train_graphs, x_extra=train_x, x_sys=train_sys,
                            x_ids=data.df.iloc[data.train_indices].smiles.tolist(),
                            y=train_y, dataset_info=data.df.dataset_id.iloc[data.train_indices].tolist(),
                            void_info=data.void_info, y_neg=True)
    trainloader = DataLoader(traindata, 32, shuffle=True)

    # trainloader = test_newbg()
    batch = list(trainloader)[0]
    x, y, weights = batch
    ranker = MPNranker(1, 9)
    ranker(x)
    from mpnranker2 import train
    # practically 5 seeds:
    for i in range(5):
        ranker = MPNranker(1, 9, [64, 16], [256], encoder_size=256)
        train(ranker, trainloader, 20, learning_rate=5e-4, margin_loss=0.2, ep_save=True)
    # acc for one batch
    ((ranker(x)[0] > 0.5).int() == y).sum() / len(y)

    data_eval(ranker, data)
    ranker.predict(*x[0])
    torch.save(ranker, 'mpnranker_newest_ep0.pt')

    # confl files
    c = {f.split('/')[-1].split('.')[0]: pickle.load(open(f, 'rb')) for f in ['../../Uni/RTpred/pairs_0068_0138.pkl',
                                                 '../../Uni/RTpred/pairs2b.pkl',
                                                 '../../Uni/RTpred/pairs2.pkl',
                                                 '../../Uni/RTpred/pairs3.pkl']}

def test_standardization():
    # train data
    data_train = Data(use_hsm=True,
                      use_ph=True)
    for did in ['0068', '0138']:
        data_train.add_dataset_id(did, isomeric=True)
    data_train.compute_features(mode=None)
    data_train.compute_graphs()
    data_train.split_data()
    print(data_train.train_x[0])
    print(data_train.train_sys[0])
    data_train.standardize()
    print(data_train.train_x[0])
    print(data_train.train_sys[0])
    # test data
    data_test = Data(use_hsm=True,
                     use_ph=True)
    for did in ['0069', '0139']:
        data_test.add_dataset_id(did, isomeric=True)
    data_test.compute_features(mode=None)
    data_test.compute_graphs()
    data_test.split_data((0., 0.))
    print(data_test.train_x[0])
    print(data_test.train_sys[0])
    desc_scaler = data_train.descriptor_scaler if hasattr(data_train, 'descriptor_scaler') else None
    sys_scaler = data_train.sysfeature_scaler if hasattr(data_train, 'sysfeature_scaler') else None
    data_test.standardize(other_descriptor_scaler=desc_scaler, other_sysfeature_scaler=sys_scaler,
                          can_create_new_scaler=False)
    print(data_test.train_x[0])
    print(data_test.train_sys[0])
