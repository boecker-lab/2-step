from random import shuffle
from mpnranker2 import MPNranker
from evaluate import load_model
import torch
from utils import Data
from features import parse_feature_spec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from itertools import combinations

def guess_from_data(d: Data):
    possibilities = set(list(map(tuple, np.concatenate([d.train_x, d.val_x, d.test_x])[:, 1:].tolist())))
    yield from possibilities

def simple_loss_fun(x1, x2, y):
    return ((-y * (x1 - x2)) > 0).numpy().astype(int)


def try_column_vector(m: MPNranker, bg, guess, mpn_margin=0.1):
    # loss_fun = torch.nn.MarginRankingLoss(mpn_margin, reduction='none')
    # loss_fun = torch.nn.BCELoss(reduction='none')
    loss_fun = simple_loss_fun
    m.eval()
    with torch.no_grad():
        losses = []
        for batch in bg:
            x, y, weights = batch
            y = torch.as_tensor(y).float().to(m.encoder.device)
            # weights = torch.as_tensor(weights).to(m.encoder.device)
            x[0][1] = torch.as_tensor(x[0][1]).float().to(m.encoder.device)
            x[1][1] = torch.as_tensor(x[1][1]).float().to(m.encoder.device)
            cf_guess = torch.stack([torch.Tensor(guess)] * len(y))
            pred = m(((x[0][0], torch.cat([x[0][1], cf_guess], 1)),
                      (x[1][0], torch.cat([x[1][1], cf_guess], 1))))
            pickle.dump(((x[0][0], torch.cat([x[0][1], cf_guess], 1)),
                         (x[1][0], torch.cat([x[1][1], cf_guess], 1))), open('/tmp/torchvizdump.pkl', 'wb'))
            # print(pred)
            # print(pred[0] - pred[1])
            # print(y)
            # return
            loss = (loss_fun(*pred, y) * weights).tolist()
            losses.extend(loss)
    return pd.DataFrame({'loss': losses})

def try_column_vector_pred(m, x, guess):
    m.eval()
    with torch.no_grad():
        x[1] = torch.as_tensor(x[1]).float().to(m.encoder.device)
        pickle.dump(((x[0], torch.cat([x[1], torch.stack([torch.Tensor(guess)] * len(x[1]))], 1)), ), open('/tmp/torchvizdump.pkl', 'wb'))
        return m(((x[0], torch.cat([x[1], torch.stack([torch.Tensor(guess)] * len(x[1]))], 1)), ))

def guesses_different_datasets(dss, ranker, guesses, scaler, custom_features=['MolLogP']):
    losses = []
    for ds in tqdm(dss):
        data_args = {'use_system_information': False,
                     'metadata_void_rt': True,
                     'custom_features': custom_features,
                     'use_hsm': False,
                     'custom_column_fields': None,
                     'columns_remove_na': False,
                     'graph_mode': True,
                     'repo_root_folder': '/home/lo63tor/rtpred/repo_newest'}
        test_data = Data(**data_args)
        test_data.add_dataset_id(ds, isomeric=True, repo_root_folder='/home/lo63tor/rtpred/repo_newest')
        test_data.compute_features(mode=parse_feature_spec('rdkall')['mode'])
        test_data.compute_graphs()
        test_data.split_data((0, 0))
        test_data.standardize(scaler)
        ((train_graphs, train_x, train_y),
         (val_graphs, val_x, val_y),
         (test_graphs, test_x, test_y)) = test_data.get_split_data()
        X = np.concatenate((train_x, test_x, val_x))
        graphs = np.concatenate((train_graphs, test_graphs, val_graphs))
        Y = np.concatenate((train_y, test_y, val_y))
        bg = BatchGenerator((train_graphs, train_x), train_y,
                            ids=test_data.df.iloc[test_data.train_indices].smiles.tolist(),
                            dataset_info=test_data.df.dataset_id.iloc[test_data.train_indices].tolist(),
                            void_info=test_data.void_info,
                            multix=True, y_neg=True)
        losses.extend([{'guess': i, 'loss': try_column_vector(ranker, bg, guess).loss.mean(),
                       'dataset': ds}
                       for i, guess in enumerate(guesses)])
    return pd.DataFrame.from_records(losses).set_index(['dataset', 'guess'])

if __name__ == '__main__':
    import os.path
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--mode', choices=['testall', 'specific', 'allpairs'])
    parser.add_argument('--pairs_file', default='/home/fleming/Documents/Uni/RTpred/pairs6.pkl')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', action='store_true')
    # args = parser.parse_args('--mode allpairs runs/graphformer/graphformer_small_ep10'.split())
    args = parser.parse_args()
    m, data, config = load_model(args.model, 'mpn')
    if (args.gpu):
        torch.set_default_device('cuda')
    else:
        if hasattr(m, 'encoder') and hasattr(m.encoder, 'device'):
            m.device = m.encoder.device = m.encoder.encoder[0].device = torch.device('cpu')
    if (args.mode == 'testall'):
        # test all column param configurations
        test = "0048 0072 0078 0084 0098 0083 0076 0101 0019 0079 0099 0070 0102 0087 0086 0066 0062 0017 0095 0067 0097 0082 0124 0069 0181 0024 0085 0093 0094 0100 0092 0179 0068".split()
        losses = guesses_different_datasets(['0226'] + test, m, list(guess_from_data(data)), data.scaler)
        losses.to_csv(f'guess_columnparams_losses_{os.path.splitext(os.path.basename(args.model))[0]}.tsv', sep='\t')
    elif (args.mode == 'specific'):
        # specific data
        df = data.df.reset_index()
        # get caffeine/phenylalanine pair on datasets 0067 and 0002 (conflicting)
        df.loc[~df.name.isna() & (df.name.str.contains('caffeine', case=False) | df.name.str.contains('phenylalanine', case=False)) & (df.id.str.contains('0067_') | df.id.str.contains('0002_'))]
        indices = {
            ('c', '0067'): df.loc[~df.name.isna() & (df.name.str.contains('caffeine', case=False)) & (df.id.str.contains('0067_'))].index[0],
            ('c', '0002'): df.loc[~df.name.isna() & (df.name.str.contains('caffeine', case=False)) & (df.id.str.contains('0002_'))].index[0],
            ('F', '0067'): df.loc[~df.name.isna() & (df.name.str.contains('phenylalanine', case=False) & (df.formula == 'C9H11NO2')) & (df.id.str.contains('0067_'))].index[0],
            ('F', '0002'): df.loc[~df.name.isna() & (df.name.str.contains('phenylalanine', case=False) & (df.formula == 'C9H11NO2')) & (df.id.str.contains('0002_'))].index[0]}
        m.eval()
        with torch.no_grad():
            rois = {i: m([[[data.get_graphs()[indices[i]]], torch.as_tensor([data.get_x()[indices[i]]], dtype=torch.float)]])
                    for i in indices}
        rts = {i: data.get_y()[indices[i]] for i in indices}
        from pprint import pprint
        print('(compound, dataset): (roi, rt)')
        pprint({i: (rois[i], rts[i]) for i in indices})
    elif (args.mode == 'allpairs'):
        # NOTE: only checks for pairs (with datasets) occuring in data.pkl!!!
        # make a tsv out of all conflicting pairs then use this to make Data object
        # TODO: make this batched, make predictions beforehand and only once!!
        #  get all rows occurring in
        import pickle
        df = data.df.reset_index()
        pairs = pickle.load(open(args.pairs_file, 'rb'))
        # what has to be predicted?
        to_predict = []
        for p in pairs:
            for dss in pairs[p]:
                for ds in dss:
                    c1, c2 = p
                    to_predict.append((ds, c1))
                    to_predict.append((ds, c2))
        to_predict = set(to_predict)
        to_predict_data = {'graphs': [], 'extra': [], 'sysf': []}
        to_predict_data_info = {'input': [], 'rt': []}
        data_graphs = data.get_graphs()
        data_x_features = data.get_x()[0]
        data_x_sys = data.get_x()[1]
        if (hasattr(m, 'add_sys_features') and m.add_sys_features):
            from utils_newbg import sysfeature_graph
            from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
            if (m.add_sys_features_mode == 'bond'):
                set_extra_bond_fdim(data_x_sys.shape[1])
            elif (m.add_sys_features_mode == 'atom'):
                set_extra_atom_fdim(data_x_sys.shape[1])
        for i, r in df.iterrows():
            if (input_:=(r.dataset_id, r.smiles)) in to_predict:
                if (m.add_sys_features):
                    to_predict_data['graphs'].append(sysfeature_graph(
                        r.smiles, data_graphs[i], data_x_sys[i],
                        bond_or_atom=m.add_sys_features_mode))
                else:
                    to_predict_data['graphs'].append(data_graphs[i])
                to_predict_data['extra'].append(data_x_features[i])
                to_predict_data['sysf'].append(data_x_sys[i])
                to_predict_data_info['input'].append(input_)
                to_predict_data_info['rt'].append(r.rt)
        to_predict_data['extra'] = np.array(to_predict_data['extra'], dtype=np.float32)
        to_predict_data['sysf'] = np.array(to_predict_data['sysf'], dtype=np.float32)
        # do the predictions
        print('making predictions...')
        preds = m.predict(**to_predict_data, batch_size=args.batch_size, prog_bar=True)
        preds = dict(zip(to_predict_data_info['input'], preds))
        rts = dict(zip(to_predict_data_info['input'], to_predict_data_info['rt']))
        sysf = dict(zip(to_predict_data_info['input'], map(lambda x: tuple(np.round(xi, 3) for xi in x),
                                                      to_predict_data['sysf'])))
        pairs_keys = list(pairs)
        shuffle(pairs_keys)
        correct_pairs = []
        verb_counter = 0        # for verbose print at most 50 instances
        wait_till_correct = False
        waited_till_correct = False
        for p in tqdm(pairs_keys):
            datasets = pairs[p]
            if (args.verbose and not wait_till_correct and verb_counter >= 10):
                break
            if (waited_till_correct):
                break
            for dss in datasets:
                # check if part of data
                if (not all([(ds, c) in preds for c in p for ds in dss])):
                    continue
                # check order
                c1, c2 = p
                is_really_conflicting = any([(rts[(ds1, c1)] - rts[(ds1, c2)]) * (rts[(ds2, c1)] - rts[(ds2, c2)])
                                             < 0 for ds1, ds2 in combinations(dss, 2)])
                if (not is_really_conflicting):
                    continue
                verb_counter += 1
                correct = {ds: (rts[(ds, c1)] - rts[(ds, c2)]) * (preds[(ds, c1)] - preds[(ds, c2)]) > 0
                           for ds in dss}
                assert all(map(lambda x: len(x) == 1, (sysf_p:=({ds: list(set(sysf[(ds, c)] for c in p))
                                                            for ds in dss})).values())
                           ), 'different sys features with the same dataset?'
                could_be_correct = any([not np.isclose(sysf_p[ds1], sysf_p[ds2], atol=1e-3).all()
                                        for ds1, ds2 in combinations(dss, 2)])
                if (wait_till_correct and all(correct)):
                    waited_till_correct = True
                if (args.verbose):
                    verb_df = pd.DataFrame.from_records(
                        [{'compound': c, 'dataset': ds, 'roi': preds[(ds, c)], 'rt': rts[(ds, c)],
                          'correct': correct[ds], 'could_be_correct': could_be_correct,
                          'roi_diff': np.abs(preds[(ds, c1)] - preds[(ds, c2)])}
                         for c in p for ds in dss])
                    print(verb_df.set_index(['dataset', 'compound']).sort_values(by=['dataset', 'rt']))
                correct_pairs.append({'pair': p, 'datasets': dss, 'correct': all(correct.values()),
                                      'could_be_correct': could_be_correct})
        correct_df = pd.DataFrame.from_records(correct_pairs)
        correct_df_possible = correct_df.loc[correct_df.could_be_correct]
        if (not args.no_save):
            correct_df.to_csv(f'confl_pairs_preds_{os.path.splitext(os.path.basename(args.model))[0]}.tsv', sep='\t')
        print(correct_df.groupby('pair').describe())
        print('correct pairs (for any one dataset pair):   ' +
              f'{correct_df.correct.sum()}/{len(correct_df)} ({correct_df.correct.sum()/len(correct_df):.2%})')
        print('only those that are possible to predict     ' +
              f'{correct_df_possible.correct.sum()}/{len(correct_df_possible)} ({correct_df_possible.correct.sum()/len(correct_df_possible):.2%})')
        num_correct = (correct_df.groupby('pair').correct.min() == True).sum()
        print('pairs correct for all dataset pairs:        ' +
              f'{num_correct}/{len(correct_df.groupby("pair"))} ({num_correct/len(correct_df.groupby("pair")):.2%})')
        num_correct = (correct_df_possible.groupby('pair').correct.min() == True).sum()
        print('[only those that are possible to predict]   ' +
              f'{num_correct}/{len(correct_df.groupby("pair"))} ({num_correct/len(correct_df.groupby("pair")):.2%})')
