from random import shuffle
from mpnranker2 import MPNranker
from evaluate import load_model
import torch
from utils import Data, BatchGenerator, get_column_scaling
from features import parse_feature_spec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def guess_from_data(d: Data):
    possibilities = set(list(map(tuple, np.concatenate([d.train_x, d.val_x, d.test_x])[:, 1:].tolist())))
    yield from possibilities

def simple_loss_fun(x1, x2, y):
    return ((-y * (x1 - x2)) > 0).numpy().astype(int)


def try_column_vector(m: MPNranker, bg: BatchGenerator, guess, mpn_margin=0.1):
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
                     'repo_root_folder': '/home/lo63tor/rtpred/RtPredTrainingData'}
        test_data = Data(**data_args)
        test_data.add_dataset_id(ds, isomeric=True, repo_root_folder='/home/lo63tor/rtpred/RtPredTrainingData')
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
    parser.add_argument('--pairs_file', default='/home/fleming/Documents/Uni/RTpred/pairs2.pkl')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    # args = parser.parse_args('/home/fleming/Documents/Projects/rtranknet/runs/newweightsconfl/newweightsconfl'.split())
    args = parser.parse_args()
    m, data, config = load_model(args.model, 'mpn')
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
        import pickle
        df = data.df.reset_index()
        pairs = pickle.load(open(args.pairs_file, 'rb'))
        pairs_keys = list(pairs)
        shuffle(pairs_keys)
        correct_pairs = []
        verb_counter = 0        # for verbose print at most 50 instances
        for p in tqdm(pairs_keys):
            datasets = pairs[p]
            if (args.verbose and verb_counter >= 10):
                break
            verb_counter += 1
            for dss in datasets:
                # check if part of data
                rows = {(c, ds): df.loc[(df['smiles.std'] == c) &
                                        (df.dataset_id == ds if 'dataset_id' in df.columns
                                         else df.id.str.contains(ds + '_'))].index
                        for c in p for ds in dss}
                if (not all(len(v) > 0 for v in rows.values())):
                    continue
                # for normal [(graphs, extra, sys), (graphs, extra, sys)] ranker
                rois = {i: m([[[data.get_graphs()[rows[i][0]]],
                               torch.as_tensor([data.get_x()[0][rows[i][0]]], dtype=torch.float).to(m.encoder.device),
                               torch.as_tensor([data.get_x()[1][rows[i][0]]], dtype=torch.float).to(m.encoder.device)]]
                             )[0].detach().cpu().numpy()[0]
                        for i in rows}
                # for nonsym [((graphs, extra), (graphs, extra)), sys] ranker
                # rois = {i: 1- m([([data.get_graphs()[rows[i][0]]],
                #                torch.as_tensor([data.get_x()[0][rows[i][0]]], dtype=torch.float)),
                #               torch.as_tensor([data.get_x()[1][rows[i][0]]], dtype=torch.float)]).detach().item()
                #         for i in rows}
                # pickle.dump([[[data.get_graphs()[list(rows.values())[0][0]]], torch.as_tensor([data.get_x()[list(rows.values())[0][0]]], dtype=torch.float)]], open('/tmp/torchvizdump.pkl', 'wb'))
                # check order
                if (len(p) != 2):
                    print('wtf?', p)
                    continue
                c1, c2 = p
                correct = [(data.get_y()[rows[(c1, ds)][0]] - data.get_y()[rows[(c2, ds)][0]]) * (rois[(c1, ds)] - rois[(c2, ds)]) > 0
                           for ds in dss]
                if (args.verbose):
                    # print(rois)
                    # print(correct)
                    verb_df = pd.DataFrame.from_records(
                        [{'compound': c, 'dataset': ds, 'roi': rois[(c, ds)], 'rt': data.get_y()[rows[(c, ds)][0]]}
                         for c, ds in rows])
                    print(verb_df.set_index(['dataset', 'compound']).sort_values(by=['dataset', 'rt']))
                    print(verb_df.groupby('compound').roi.diff().dropna().abs())
                    # gradient fuer ROIs
                    # rois = {i: m([[[data.get_graphs()[rows[i][0]]],
                    #            torch.as_tensor([data.get_x()[0][rows[i][0]]], dtype=torch.float),
                    #            torch.as_tensor([data.get_x()[1][rows[i][0]]], dtype=torch.float)]]
                    #          )[0][0].detach().numpy()[0]
                    #     for i in rows}
                correct_pairs.append({'pair': p, 'datasets': dss, 'correct': all(correct)})
        correct_df = pd.DataFrame.from_records(correct_pairs)
        if (not args.no_save):
            correct_df.to_csv(f'confl_pairs_preds_{os.path.splitext(os.path.basename(args.model))[0]}.tsv', sep='\t')
        print(correct_df.groupby('pair').describe())
        num_correct = len(correct_df.loc[correct_df.correct == True])
        print('correct pairs (for one dataset): '
              f'{num_correct}/{len(correct_df)} ({num_correct/len(correct_df):.2%})')
        num_correct = (correct_df.groupby('pair').correct.min() == True).sum()
        print('pairs correct for all datasets: '
              f'{num_correct}/{len(correct_df.groupby("pair"))} ({num_correct/len(correct_df.groupby("pair")):.2%})')
