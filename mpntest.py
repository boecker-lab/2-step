import pandas as pd
from mpnranker import MPNranker, train, predict
from utils import BatchGenerator
from chemprop.features import mol2graph
from tensorboardX.writer import SummaryWriter
import torch.nn as nn
import torch
import sys
import numpy as np


if __name__ == '__main__':
    train_set = pd.read_csv(
        # '/home/fleming/Documents/Projects/chemprop/tmp/all_smrt_train.csv',
        sys.argv[1],
        names=['smiles', 'rt'], header=0)
    test_set = pd.read_csv(
        # '/home/fleming/Documents/Projects/chemprop/tmp/all_smrt_test.csv',
        sys.argv[2],
        names=['smiles', 'rt'], header=0)
    # batch_size = 16384
    batch_size = int(sys.argv[3])
    sigmoid = sys.argv[4].lower() == 't'
    margin = float(sys.argv[5])
    if (len(sys.argv) > 6):
        name = sys.argv[6]
        writer = SummaryWriter(f'runs/{name}_train'); val_writer = SummaryWriter(f'runs/{name}_val')
    else:
        writer = SummaryWriter(comment='_train'); val_writer = SummaryWriter(comment='_val')
    train_df = train_set.iloc[:70_000].sample(70_000)
    # train_df = train_set.iloc[:70_000].sample(2000)
    val_df = train_set.iloc[70_000:].loc[train_set.rt > 8].sample(2000)
    print('computing mol graphs...')
    train_df['graphs'] = [mol2graph([s]) for s in train_df.smiles]
    val_df['graphs'] = [mol2graph([s]) for s in val_df.smiles]
    test_set['graphs'] = [mol2graph([s]) for s in test_set.smiles]
    print('done. Initializing BatchGenerator...')
    bg = BatchGenerator(train_df.graphs.values, train_df.rt.values,
                        pair_step=3, pair_stop=128, batch_size=batch_size,
                        void=8, y_neg=not sigmoid)
    val_g = BatchGenerator(val_df.graphs.values, val_df.rt.values,
                           pair_step=3, pair_stop=128, batch_size=batch_size,
                           void=8, y_neg=not sigmoid)
    ranker = MPNranker(sigmoid=sigmoid)
    print('done. Starting training')
    train(ranker, bg, 5, writer, val_g, val_writer=val_writer,
          steps_train_loss=np.ceil(len(bg) / 100).astype(int),
          # steps_train_loss=3,
          steps_val_loss=np.ceil(len(bg) / 5).astype(int),
          # steps_val_loss=10,
          batch_size=batch_size,
          sigmoid_loss=sigmoid, margin_loss=margin)
    train_acc, val_acc, test_acc = (predict(train_df, ranker, batch_size=batch_size),
                                    predict(val_df, ranker, batch_size=batch_size),
                                    predict(test_set, ranker, batch_size=batch_size))
    print(f'{train_acc=:.2%}, {val_acc=:.2%}, {test_acc=:.2%}')
    # writer.export_scalars_to_json('./all_scalars.json')
    writer.close(); val_writer.close()

    """
    loss_fun = nn.MarginRankingLoss(0.1)
    x, y, weights = bg[0]
    preds = ranker(x)
    loss = loss_fun(*preds, torch.Tensor([y]).transpose(0, 1))
    """
