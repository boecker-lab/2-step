import pandas as pd
from mpnranker import MPNranker, train, predict
from utils import BatchGenerator
from chemprop.features import mol2graph
from tensorboardX.writer import SummaryWriter
import torch.nn as nn
import torch
import sys


if __name__ == '__main__':
    train_set = pd.read_csv(
        # '/home/fleming/Documents/Projects/chemprop/tmp/all_smrt_train.csv',
        sys.argv[1],
        names=['smiles', 'rt'], header=0)
    test_set = pd.read_csv(
        # '/home/fleming/Documents/Projects/chemprop/tmp/all_smrt_test.csv',
        sys.argv[2],
        names=['smiles', 'rt'], header=0)

    writer = SummaryWriter(comment='_train'); val_writer = SummaryWriter(comment='_val')
    train_df = train_set.iloc[:70_000].sample(70_000)
    val_df = train_set.iloc[70_000:].loc[train_set.rt > 8].sample(1000)
    train_df['graphs'] = [mol2graph([s]) for s in train_df.smiles]
    val_df['graphs'] = [mol2graph([s]) for s in val_df.smiles]
    test_set['graphs'] = [mol2graph([s]) for s in test_set.smiles]
    bg = BatchGenerator(train_df.graphs.values, train_df.rt.values,
                        pair_step=3, pair_stop=128, batch_size=1024,
                        void=8)
    val_g = BatchGenerator(val_df.graphs.values, val_df.rt.values,
                           pair_step=3, pair_stop=128, batch_size=1024,
                           void=8)
    ranker = MPNranker()
    train(ranker, bg, 2, writer, val_g, val_writer=val_writer)
    train_acc, val_acc, test_acc = (predict(train_df, ranker),
                                    predict(val_df, ranker),
                                    predict(test_set, ranker))
    print(f'{train_acc=:.2%}, {val_acc=:.2%}, {test_acc=:.2%}')
    writer.export_scalars_to_json('./all_scalars.json')
    writer.close(); val_writer.close()

    """
    loss_fun = nn.MarginRankingLoss(0.1)
    x, y, weights = bg[0]
    preds = ranker(x)
    loss = loss_fun(*preds, torch.Tensor([y]).transpose(0, 1))
    """
