from mpnranker import MPNranker
from evaluate import load_model
import torch
from utils import Data, BatchGenerator, get_column_scaling
from features import parse_feature_spec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    m, data, config = load_model('runs/newweightsconfl_only_bal_ep16', 'mpn')
    test = "0048 0072 0078 0084 0098 0083 0076 0101 0019 0079 0099 0070 0102 0087 0086 0066 0062 0017 0095 0067 0097 0082 0124 0069 0181 0024 0085 0093 0094 0100 0092 0179 0068".split()
    losses = guesses_different_datasets(['0226'] + test, m, list(guess_from_data(data)), data.scaler)
    losses.to_csv('gues_columnparams_losses_newweightsconfl_only_bal_ep16.tsv', sep='\t')
