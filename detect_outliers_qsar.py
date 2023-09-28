import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from tap import Tap
from typing import List, Optional, Literal, Tuple, Union
import json
from tqdm import tqdm
import subprocess

class QSARArgs(Tap):
    out_prefix: str
    iqr_mod: float = 1.5
    repo_root_folder: str = '/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/'
    void_factor: float = 2


def get_ds_data(ds, repo_root_folder, t0=0, ret_smiles=False, void_factor=2):
    # NOTE: for duplicate indices (!) isomeric data overwrites canonical data
    rt_data_dfs = []
    descriptor_dfs = []
    for mode in ['canonical', 'isomeric']:
        try:
            rt_data_dfs.append(pd.read_csv(repo_root_folder + f'/processed_data/{ds}/{ds}_rtdata_{mode}_success.txt', sep='\t'))
        except Exception as e:
            print(e)
        try:
            descriptor_dfs.append(pd.read_csv(repo_root_folder + f'/processed_data/{ds}/{ds}_descriptors_{mode}_success.txt', sep='\t'))
        except Exception as e:
            print(e)
    rt_data = pd.concat(rt_data_dfs).drop_duplicates(subset='id', keep='last').set_index('id').sort_index()
    descriptors = pd.concat(descriptor_dfs).drop_duplicates(subset='id', keep='last').set_index('id').sort_index()
    relevant_indices = rt_data.loc[rt_data.rt > void_factor * t0].index.tolist() # also filters NaNs
    if (not ret_smiles):
        return descriptors.dropna(axis=1).loc[relevant_indices], rt_data.loc[relevant_indices, 'rt']
    else:
        return descriptors.dropna(axis=1).loc[relevant_indices], rt_data.loc[relevant_indices, 'rt'], rt_data.loc[relevant_indices, 'smiles.std']

def get_column_t0(ds, repo_root_folder):
    return pd.read_csv(repo_root_folder + f'/processed_data/{ds}/{ds}_metadata.txt', sep='\t',
                       index_col=0)['column.t0'].iloc[0]


def get_outliers(ds, repo_root_folder, regressor=RandomForestRegressor(), iqr_mod=1.5, print_errors=True,
                 print_filterered_perc=True, boxplot=False, only_errors=False, void_factor=2,
                 extra_data=False):
    extra_data_ret = {}
    t0 = get_column_t0(ds, repo_root_folder=repo_root_folder)
    X, y = get_ds_data(ds, repo_root_folder=repo_root_folder,
                       t0=t0, void_factor=void_factor)
    if (only_errors):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(y_test)
        errors = {'MAE': (y_test - y_pred).abs().mean(), 'MedAE': (y_test - y_pred).abs().median()}
        outlier_indices = []
    else:
        y_pred = cross_val_predict(regressor, X, y, n_jobs=16, cv=KFold(10, shuffle=True))
        errors = {'MAE': (y - y_pred).abs().mean(), 'MedAE': (y - y_pred).abs().median()}
        if (print_errors):
            print(', '.join(f'{e}: {errors[e]:.2f}' for e in errors))
        if (boxplot):
            (y - y_pred).abs().plot.box()
            plt.show()
        q1, q3 = (y - y_pred).abs().quantile([0.25, 0.75])
        iqr = q3 - q1
        fence_high = q3 + (iqr_mod * iqr)
        outlier_indices = y.loc[(y - y_pred).abs() > fence_high].index.tolist()
        if (print_filterered_perc):
            print(f'filtered: {len(outlier_indices) / len(y):.0%}')
        if (extra_data):
            extra_data_ret['predictions'] = pd.DataFrame({'rt_true': y, 'rt_pred': y_pred})
            extra_data_ret['void_thr'] = void_factor * t0
            extra_data_ret['error_thr'] = fence_high
    return outlier_indices, errors, extra_data_ret

if __name__ == '__main__':
    args = QSARArgs().parse_args()
    import sys; sys.path.append(args.repo_root_folder); from pandas_dfs import get_data_df, get_dataset_df, order_acc, get_confl; cs_df = get_data_df(); dss = get_dataset_df()
    rp_datasets = [ds for ds in dss.loc[dss['method.type'] == 'RP'].index.tolist()
                   # only if they have >= 100 compounds
                   if cs_df.loc[(cs_df.dataset_id == ds) & (cs_df.rt > args.void_factor * dss.loc[ds, 'column.t0']),
                                'smiles.std'].nunique() >= 100
                   # not the huge SMRT dataset
                   and ds != '0186']
    outliers = {}
    errors = []
    for ds in tqdm(rp_datasets):
        print(ds + ' ...')
        try:
            regressor = GradientBoostingRegressor(n_estimators=1000, max_depth=2)
            outliers[ds], errors_current = get_outliers(
                ds, repo_root_folder=args.repo_root_folder, iqr_mod=args.iqr_mod, void_factor=args.void_factor,
                regressor=regressor)
            errors.append(errors_current)
        except Exception as e:
            print(e)
    errors_df = pd.DataFrame.from_records(errors)
    errors_df['ds'] = rp_datasets
    print(errors_df.describe())
    with open(args.out_prefix + '.txt', 'w') as out:
        out.write('id\n')
        out.write('\n'.join([u for v in outliers.values() for u in v]) + '\n')
    errors_df.to_csv(args.out_prefix + '_errors.csv')
    json.dump({'args': args._log_all(),
               'regressor': {'repr': repr(regressor), 'params': regressor.get_params()},
               'datasets': rp_datasets,
               'rtrepo_state': subprocess.run(
                   ['git', 'rev-parse',  '--short', 'HEAD'], cwd=args.repo_root_folder, capture_output=True
               ).stdout.decode().strip()},
              open(f'{args.out_prefix}_info.json', 'w'), indent=2)
