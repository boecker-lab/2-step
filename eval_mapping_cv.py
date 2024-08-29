from argparse import ArgumentParser
import sys
import pandas as pd
from mapping import LADModel
from sklearn.model_selection import KFold, ShuffleSplit
import re

def anchors_split(n_anchors=15):
    return ShuffleSplit

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('tsvs', nargs='+')
    parser.add_argument('--void_factor', default=2, type=float)
    parser.add_argument('--repo_root', default='/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/')
    parser.add_argument('--anchors', default=None, type=int)
    parser.add_argument('--folds', default=10, type=int)
    args = parser.parse_args()

    sys.path.append(args.repo_root)
    from pandas_dfs import get_dataset_df
    dss = get_dataset_df()

    records = []
    for f in args.tsvs:
        ds_id, fold, split = re.match(r'.*_(\d{4})(_fold\d+)?(_test|_train)?.tsv', f.split('/')[-1]).groups()
        if split is not None:
            raise Exception('train/test split files are not supported here, use `plot_exported.py`')
        df = pd.read_csv(f, sep='\t', names=['smiles', 'rt', 'roi'], header=None)
        df['roi2'] = df.roi ** 2 # for LAD model
        void_t = dss['column.t0'].loc[ds_id] * args.void_factor
        data_novoid = df.loc[df.rt > void_t]
        records_i = []
        if (args.anchors is not None):
            split_fun = ShuffleSplit(n_splits=args.folds, train_size=args.anchors).split
        else:
            split_fun = KFold(n_splits=args.folds, shuffle=True).split
        for i, (train_index, test_index) in enumerate(split_fun(data_novoid)):
            data_train = data_novoid.iloc[train_index]
            data_test = data_novoid.iloc[test_index]
            print(f'{i=}, {len(data_train)=}, {i=}, {len(data_test)=}')
            model = LADModel(data_train, ols_after=True, ols_discard_if_negative=True, ols_drop_mode='2*median')
            error_test = (model.get_mapping(data_test.roi) - data_test.rt).abs()
            error_train = (model.get_mapping(data_train.roi) - data_train.rt).abs()
            records_i.append(dict(MAE=error_test.mean(), MAE_train=error_train.mean(),
                                  MedAE=error_test.median(), MedAE_train=error_train.median()))
        df_i = pd.DataFrame.from_records(records_i)
        records.append(dict(ds=ds_id,
                            MAE=df_i.MAE.mean(), MAE_std=df_i.MAE.std(),
                            MAE_train=df_i.MAE_train.mean(), MAE_train_std=df_i.MAE_train.std(),
                            MedAE=df_i.MedAE.mean(), MedAE_std=df_i.MedAE.std(),
                            MedAE_train=df_i.MedAE_train.mean(), MedAE_train_std=df_i.MedAE_train.std()))
    df = pd.DataFrame.from_records(records)
    df['print'] = [f'{r["MAE"]:.3f}±{r["MAE_std"]:.3f}' for i, r in df.iterrows()]
    print(df.to_string())
