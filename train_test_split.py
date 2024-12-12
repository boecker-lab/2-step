from tap import Tap
from typing import List, Literal, Optional, Union
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, LeavePGroupsOut, KFold, GroupShuffleSplit
from glob import glob
import os.path
from random import sample
from itertools import combinations

FAULTY = ["0006", "0008", "0023", "0024", "0027", "0046", "0056", "0057", "0059", "0103", "0104", "0105", "0107", "0110",
          "0111", "0112", "0113", "0114", "0116", "0117", "0118", "0119", "0120", "0123", "0128",
          "0130", "0131", "0132", "0133", "0136", "0137", "0139", "0143", "0145", "0148", "0149",
          "0151", "0152", "0154", "0155", "0156", "0157", "0160", "0161", "0162", "0163", "0164",
          "0165", "0166", "0167", "0168", "0169", "0170", "0171", "0172", "0173", "0174", "0175",
          "0176", "0177", "0178"]

class SplittingArgs(Tap):
    run_name: str
    mode: Literal['random', 'study', 'setup', 'setup2', 'column', 'columnph', 'columnphdiff', 'columnphdiff2', 'columnorphdiff', 'all'] = 'random'
    dataset_blacklist: Optional[List[str]] = ['0186']  + FAULTY # exclude SMRT and faulty list by default
    dataset_whitelist: Optional[List[str]] = None
    hsm_required: bool = False
    tanaka_required: bool = False
    ph_required: bool = False
    columnname_required: bool = False
    gradient_required: bool = False
    setup_cluster_fields: List[str] = ['column.name', 'column.length', 'column.id', 'column.particle.size',
                                       'column.temperature', 'column.flowrate', 'mobilephase']
    k_fold: int = 5
    repo_root_folder: str = '../RepoRT/'
    test_arg: List[str] = []
    nr_initial_datasets: int = -1
    out_dir: Optional[str] = None

def get_gradient_missing_sets(repo_root_folder, verbose=True):
    sets = []
    for dir_ in glob(f'{repo_root_folder}/processed_data/*/'):
        ds = dir_.split('/')[-2]
        gd = dir_ + ds + '_gradient.tsv'
        try:
            if not os.path.exists(gd) or len(pd.read_csv(gd, sep='\t').dropna()) == 0:
                if (verbose):
                    print(ds, 'missing', gd)
                sets.append(ds)
        except:
            if (verbose):
                print(ds, 'error', gd)
            sets.append(ds)
    return sets

def get_conflicting_groups(initial_groups):
    conflicting_groups = {}
    for (ds_id1, (group1, column1, ph1)), (ds_id2, (group2, column2, ph2)) in combinations(initial_groups.items(), 2):
        if (group1 == group2) or (column1 != column2):
            continue
        ph_diff = np.abs(ph1 - ph2)
        if ph_diff < 1:
            conflicting_groups.setdefault(group1, set()).add(group2)
            conflicting_groups.setdefault(group2, set()).add(group1)
    return conflicting_groups

def resolve_conflicting_groups(conflicting_groups):
    new_groups = {}
    for group1, other_groups in sorted(conflicting_groups.items(), key=lambda x: len(x[1]), reverse=True):
        all_groups = [group1] + list(other_groups)
        for g in all_groups:
            if g in new_groups:
                new_group = new_groups[g]
                break
        else:
            new_group = group1.split(';')[0] + ';' + '_'.join(sorted([_.split(';')[1] for _ in all_groups],
                                                                     key=float))
        for group in all_groups:
            new_groups[group] = new_group
    return new_groups

if __name__ == '__main__':
    args = SplittingArgs().parse_args()
    # args = SplittingArgs().parse_args('--run_name test'.split())
    if (len(args.test_arg) > 0):
        print(args.test_arg)
        exit(0)
    sys.path.append(args.repo_root_folder)
    try:
        from pandas_dfs import get_dataset_df, get_data_df
    except ImportError:
        print(f'ERROR: make sure `pandas_dfs.py` is present and current in {args.repo_root_folder}')
        exit(-1)
    dss = get_dataset_df()
    cs_df = get_data_df()
    if (args.gradient_required):
        args.dataset_blacklist = (args.dataset_blacklist or []) + get_gradient_missing_sets(args.repo_root_folder)
    dss['nr_compounds'] = cs_df.dropna(subset=['smiles.std', 'rt']).groupby('dataset_id')['smiles.std'].nunique()
    gradient_columns = [c for c in dss.columns if c.startswith('gradient.')
                        and not c.endswith('.unit')] # TODO: has to be changed once "unit" works properly
    eluent_columns = [c for c in dss.columns if c.startswith('eluent.')
                      and not (c.endswith('.unit') or c.endswith('.pH'))] # TODO: has to be changed once "unit" works properly
    dss['eluent'] = [', '.join([f'{c}:{dss.loc[s, c]}' for c in eluent_columns
                                if not isinstance(dss.loc[s, c], str) and
                                not pd.isna(dss.loc[s, c]) and dss.loc[s, c] > 0])
                     for s in dss.index.tolist()]
    dss['gradient'] = [', '.join([f'{c}:{dss.loc[s, c]}' for c in gradient_columns
                                  if not isinstance(dss.loc[s, c], str) and
                                  not pd.isna(dss.loc[s, c]) and dss.loc[s, c] > 0])
                       for s in dss.index.tolist()]
    dss['mobilephase'] = [', '.join([f'{c}:{dss.loc[s, c]}' for c in eluent_columns + gradient_columns
                                     if not isinstance(dss.loc[s, c], str) and
                                     not pd.isna(dss.loc[s, c]) and dss.loc[s, c] > 0])
                          for s in dss.index.tolist()]
    sets = dss.loc[(dss['method.type'] == 'RP') & (dss.nr_compounds > 20)].index.tolist()
    if (args.dataset_whitelist is not None):
        sets = [s for s in sets if s in args.dataset_whitelist]
        if (len(sets) < len(args.dataset_whitelist)):
            print('WARNING: not all sets in the whitelist were found:\n'
                  f'{", ".join(sorted(set(args.dataset_whitelist) - set(sets)))}'
                  ' are missing')
    if (args.dataset_blacklist is not None):
        sets = [s for s in sets if s not in args.dataset_blacklist]
    if (args.hsm_required):
        sets = dss.loc[sets].loc[~pd.isna(dss.loc[sets, 'H'])].index.tolist()
    if (args.tanaka_required):
        sets = dss.loc[sets].loc[~pd.isna(dss.loc[sets, 'αCH2'])].index.tolist()
    if (args.ph_required):
        sets = dss.loc[sets].loc[~pd.isna(dss.loc[sets, 'ph'])].index.tolist()
    if (args.columnname_required):
        sets = dss.loc[sets].loc[~pd.isna(dss.loc[sets, 'column.name'])].index.tolist()
    if (args.nr_initial_datasets != -1):
        sets = sample(sets, args.nr_initial_datasets)
    # make two cluster sets: 1) author-based, 2) setup-based
    clusters_author = dss.loc[sets].groupby('authors').agg({'id':lambda x: list(x)})
    clusters_list_author = clusters_author['id'].tolist()
    clusters_list_author_left = [s for s in sets if s not in {ci for cluster in clusters_list_author
                                                              for ci in cluster}]
    print(f'datasets clustered into {len(clusters_list_author)} author-based clusters '
          f'({len(clusters_list_author_left)} datasets without author information)')
    fields = args.setup_cluster_fields
    dss['setup'] = dss[fields].astype(str).agg(';'.join, axis=1)
    dss['columnph'] = dss[['column.name', 'ph']].astype(str).agg(';'.join, axis=1)
    # making groups for ph-difference is much more complicated:
    # first, group by columnph, then make extra groups when ph diff <= 1
    initial_groups = {i: (r['columnph'], r['column.name'], r['ph'])
                      for i, r in dss.iterrows()
                      if not pd.isna(r['column.name']) and not pd.isna(r['ph'])}
    conflicting_groups = get_conflicting_groups(initial_groups)
    new_groups = resolve_conflicting_groups(conflicting_groups)
    resolved_groups = {k: (new_groups.get(v[0], v[0]), v[1], v[2]) for k, v in initial_groups.items()}
    assert len(new_resolved:=get_conflicting_groups(resolved_groups)) == 0, ('resolving groups unsuccessful!', new_resolved)
    dss['columnphdiff'] = [resolved_groups.get(ds_id, (np.nan, ))[0] for ds_id in dss.index.tolist()]
    print(dss.setup.to_string())
    clusters_setup = dss.loc[sets].groupby(fields).agg({'id':lambda x: list(x)})
    clusters_list_setup = clusters_setup['id'].tolist()
    clusters_list_setup_left = [s for s in sets if s not in {ci for cluster in clusters_list_setup
                                                             for ci in cluster}]
    print(f'datasets clustered into {len(clusters_list_setup)} setup-based clusters '
          f'({len(clusters_list_setup_left)} datasets without sufficient setup information)')

    split_gen, field = {'random': (KFold(args.k_fold, shuffle=True).split(sets), 'authors'),
                        # 'study': LeavePGroupsOut(2).split(sets, groups=dss.loc[sets, 'authors'].tolist()),
                        # 'study': KFold(args.k_fold).split(sets, groups=dss.loc[sets, 'authors'].tolist()),
                 'study': (GroupShuffleSplit(args.k_fold, test_size=0.2).split(
                     sets, groups=dss.loc[sets, 'authors'].tolist()), 'authors'),
                 'setup': (GroupShuffleSplit(args.k_fold, test_size=0.2).split(
                     sets, groups=dss.loc[sets, 'setup'].tolist()), 'setup'),
                 'setup2': (GroupKFold(args.k_fold).split(
                     sets, groups=dss.loc[sets, 'setup'].tolist()), 'setup'),
                 # 'setup': KFold(args.k_fold).split(sets, groups=dss.loc[sets, 'setup'].tolist()),
                 'column': (GroupShuffleSplit(args.k_fold, test_size=0.2).split(
                     sets, groups=dss.loc[sets, 'column.name'].tolist()), 'column.name'),
                 # 'column': KFold(args.k_fold).split(sets, groups=dss.loc[sets, 'column.name'].tolist())
                 'columnph': (GroupShuffleSplit(args.k_fold, test_size=0.2).split(
                     sets, groups=dss.loc[sets, 'columnph'].tolist()), 'columnph'),
                 'columnphdiff': (GroupShuffleSplit(args.k_fold, test_size=0.2).split(
                     sets, groups=dss.loc[sets, 'columnphdiff'].tolist()), 'columnphdiff'),
                 'columnphdiff2': (GroupKFold(args.k_fold).split(
                     sets, groups=dss.loc[sets, 'columnphdiff'].tolist()), 'columnphdiff'),
                 'all': ([(range(len(sets)), [])], 'authors'),
                 }[args.mode]
    prefix = (args.out_dir or '') + f'{args.run_name}_fold'
    pd.options.display.max_rows = len(sets)
    for i, (train, test) in enumerate(split_gen):
        train_sets = [sets[_] for _ in train]
        test_sets = [sets[_] for _ in test]
        print('='*30 + f' fold {i} ' + '='*30)
        print(dss.loc[train_sets, ['nr_compounds', field]])
        print(dss.loc[test_sets, ['nr_compounds', field]])
        # print(' '.join(train_sets))
        # print(' '.join(test_sets))
        with open(f'{prefix}_{i+1}_train.txt', 'w') as out:
            out.write(' '.join(train_sets) + '\n')
        with open(f'{prefix}_{i+1}_test.txt', 'w') as out:
            out.write(' '.join(test_sets) + '\n')
