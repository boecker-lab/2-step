from itertools import chain
from glob import glob
from argparse import ArgumentParser
import re
import json
import pandas as pd
import numpy as np

BENCHMARK_DATASETS = ['0003', '0010', '0018', '0055', '0054', '0019', '0002']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('eval_jsons', help='*_eval.json files', nargs='+')
    parser.add_argument('--best_epoch_over', help='best epoch determined by test or train datasets',
                        choices=['train', 'test'], default='train')
    parser.add_argument('--always_last_epoch', help='don\'t determine best epoch, just use the last',
                        action='store_true')
    parser.add_argument('--final_accs_over', help='final stats determined by test or train datasets',
                        choices=['train', 'test'], default='test')
    parser.add_argument('--splits_dir', help='directory containing dataset-split files',
                        default='/home/fleming/Documents/Projects/rtranknet/splits/FE')
    parser.add_argument('--splits_prefixes', help='filename prefixes for dataset-split files',
                        default=['setup_disjoint_fold_', 'columnph_disjoint_fold_', 'columnphdiff_disjoint_fold_'], nargs='+')
    parser.add_argument('--ignore_missing_datasets', help='make stats even when datasets are missing from eval',
                        action='store_true')
    parser.add_argument('--print_fold_accs', help='prints stats for individual folds',
                        action='store_true')
    parser.add_argument('--print_all_datasets', help='prints stats for all datasets',
                        action='store_true')
    parser.add_argument('--final_metric', default='acc')

    # args = parser.parse_args('--best_epoch_over train --final_accs_over test '.split()
    #                          + list(glob('/home/fleming/Documents/Projects/rtranknet/runs/FE_sys/FE_columnph_disjoint_sys_no_fold*_ep*_eval.json')))
    args = parser.parse_args()


    # 0a. get dataset splits
    splits = {}
    for f in chain(*(glob(f'{args.splits_dir}/{splits_prefix}*_*.txt')
                     for splits_prefix in args.splits_prefixes)):
        scenario = f.split('/')[-1].split('_')[0]
        fold = int(re.search(r'.*_fold_(\d+)_.*', f).group(1))
        split_type = f.split('/')[-1].split('_')[-1].split('.')[0]
        d = splits.setdefault((scenario, fold), {})
        for ds in open(f).readlines()[0].strip().split():
            d[ds] = split_type

    # 0b. get accuracies
    dfs = []
    for f in args.eval_jsons:
        data = json.load(open(f))
        scenario = f.split('/')[-1].split('_')[1]
        fold = int(re.search(r'.*_fold(\d+)_.*', f).group(1))
        df = pd.DataFrame.from_records([{'ds':k , 'acc': data[k]['acc'],
                                         'raw_name': re.search('^.*/(.*)_eval(.*)?.json$', f).group(1),
                                         'scenario': scenario,
                                         'fold': fold,
                                         'epoch': int(re.search(r'.*_ep(\d+)_.*', f).group(1)),
                                         'ds_split': splits[(scenario, fold)].get(k, None),
                                         'benchmark': k in BENCHMARK_DATASETS} | ({args.final_metric: data[k].get(args.final_metric, np.nan)}
                                                                                  if args.final_metric != 'acc' else {})
                                        for k in data])
        dfs.append(df)
    accs = pd.concat(dfs)

    assert accs.scenario.nunique() == 1, f'only files from one scenario are supported here! found {accs.scenario.unique()}'
    scenario = accs.scenario.unique().item()

    # 1. Get best epoch per fold on `best_epoch_over` datasets
    if (not args.always_last_epoch):
        best_epoch_subset = accs.loc[accs.ds_split == args.best_epoch_over].copy().reset_index()
        best_epoch_datasets = best_epoch_subset.groupby(['fold', 'epoch']).ds.agg(list).reset_index()
        best_epoch_datasets['has_all_datasets'] = [set(r.ds) == set(k for k, v in splits[(scenario, r.fold)].items() if v == args.best_epoch_over)
                                                   for i, r in best_epoch_datasets.iterrows()]
        if (not args.ignore_missing_datasets):
            assert best_epoch_datasets.has_all_datasets.all(), f'[best_epoch_over] datasets are missing for {(~(best_epoch_datasets.has_all_datasets)).sum()} runs'

        best_epoch_subset['mean_epoch_acc'] = best_epoch_subset.groupby(['fold', 'epoch']).acc.transform('mean')
        best_epoch_subset['median_epoch_acc'] = best_epoch_subset.groupby(['fold', 'epoch']).acc.transform('median')
        best_epochs = dict(best_epoch_subset.loc[best_epoch_subset.groupby(['fold'])['mean_epoch_acc'].idxmax(), ['fold', 'epoch']].values)
    else:
        best_epochs = {fold: sorted(accs.epoch.unique())[-1] for fold in accs.fold.unique()}

    # 2. With this get metrics over on `final_accs_over` datasets
    final_accs_subset = accs.loc[accs.ds_split == args.final_accs_over].copy().reset_index()
    final_accs_subset_datasets = final_accs_subset.groupby(['fold', 'epoch']).ds.agg(list).reset_index()
    final_accs_subset_datasets['has_all_datasets'] = [set(r.ds) == set(k for k, v in splits[(scenario, r.fold)].items() if v == args.final_accs_over)
                                             for i, r in final_accs_subset_datasets.iterrows()]
    if (not args.ignore_missing_datasets):
        assert final_accs_subset_datasets.has_all_datasets.all(), f'[final_accs_over] datasets are missing for {(~(final_accs_subset_datasets.has_all_datasets)).sum()} runs'
    final_accs_subset = final_accs_subset[final_accs_subset[['fold', 'epoch']].apply(
        lambda x : best_epochs[x.fold] == x.epoch, axis=1)]
    if (args.print_all_datasets):
        print(final_accs_subset.sort_values(['fold', 'ds']))

    fold_accs = final_accs_subset.groupby(['fold'])[args.final_metric].agg(['mean', 'median', 'std']).sort_index()
    if (args.print_fold_accs):
        print(fold_accs)
    final_accs = fold_accs.agg(['mean', 'median', 'std'])
    print(f'{final_accs.loc["mean", "mean"]:.3f}±{final_accs.loc["std", "mean"]:.3f}')
