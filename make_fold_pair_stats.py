import re
import pandas as pd
from argparse import ArgumentParser
import pickle
from itertools import combinations
import pandas as pd
from tqdm import tqdm
from evaluate import get_pair_stats, get_pair_order

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_file')
    parser.add_argument('--fold_files', nargs='+')
    parser.add_argument('--scenario', default='columnph')
    parser.add_argument('--repo_root', default='/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/')
    parser.add_argument('--conflicting_smiles_pairs', default='/home/fleming/Documents/Uni/RTpred/pairs6.pkl')
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--qualifiers', default=['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)', 'kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1', 'ph'],
                        nargs='+', type=str)
    args = parser.parse_args('--fold_file /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_1_test.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_2_test.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_3_test.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_4_test.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_5_test.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_1_train.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_2_train.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_3_train.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_4_train.txt /home/fleming/Documents/Projects/rtranknet/setup_disjoint_fold_5_train.txt --scenario setup --epsilon 0.166666666667 --out_file /tmp/bla.tsv'.split())
    args = parser.parse_args()

    splits = {}
    for f in args.fold_files:
        scenario = f.split('/')[-1].split('_')[0]
        fold = int(re.search(r'.*_fold_(\d+)_.*', f).group(1))
        split_type = f.split('/')[-1].split('_')[-1].split('.')[0]
        d = splits.setdefault((scenario, fold), {})
        for ds in open(f).readlines()[0].strip().split():
            d[ds] = split_type

    import sys; sys.path.append(args.repo_root)
    from pandas_dfs import get_data_df, get_dataset_df
    cs_df = get_data_df()
    dss = get_dataset_df()

    void_info = dict(dss['column.t0'] * 2)
    epsilon = args.epsilon
    confl_pairs = pickle.load(open(args.conflicting_smiles_pairs, 'rb'))
    orig_informative_pairs = set(confl_pairs)
    records = []
    for fold in range(1, 6):
        train_sets = []
        test_sets = []
        for ds, split_type in splits[(args.scenario, fold)].items():
            if split_type == 'train':
                train_sets.append(ds)
            else:
                test_sets.append(ds)
        relevant_sets = set(train_sets) | set(test_sets)
        confl_pairs_rel = {k: {ds_pair for ds_pair in v if all(ds in relevant_sets for ds in ds_pair)}
                           for k, v in confl_pairs.items()}
        confl_pairs_rel = {k: v for k, v in confl_pairs_rel.items() if len(v) > 0}
        df_fold_train = cs_df.loc[cs_df.dataset_id.isin(train_sets)].copy()
        df_fold_train['split_type'] = 'train'
        train_compounds = set(df_fold_train['smiles.std'].unique())
        for ds_target in test_sets:
            df_fold_test = cs_df.loc[cs_df.dataset_id == ds_target].copy()
            df_fold_test['split_type'] = 'test'
            df_fold = pd.concat([df_fold_train, df_fold_test]).join(dss, how='left', on='dataset_id', rsuffix='_dss')
            stats = get_pair_stats(df_fold, ds_target=ds_target, qualifiers=args.qualifiers, confl_pairs=confl_pairs_rel, void_info=void_info, epsilon=epsilon)
            statsd = {frozenset([r.s1, r.s2]): r for i, r in stats.iterrows()}
            rts = {ds_target: {r['smiles.std']: r.rt for i, r in df_fold_test.iterrows()}}
            for s1, s2 in combinations(rts[ds_target].keys(), 2):
                order = get_pair_order(s1, s2, ds_target, rts, void_info[ds_target], epsilon=epsilon)
                if order is None:
                    continue
                p = frozenset([s1, s2])
                r = dict(fold=fold, ds=ds_target, s1=s1, s2=s2)
                if s1 not in train_compounds or s2 not in train_compounds:
                    records.append(r | dict(kind='unique_compound')) # 1 grey
                elif p not in orig_informative_pairs:
                    records.append(r | dict(kind='uninformative_general')) # 2 black
                elif p not in statsd:
                    records.append(r | dict(kind='uninformative_fold')) # 3 other black
                else:
                    row = statsd[p]
                    if row.target_ds_contradictory:
                        records.append(r | dict(kind='contradictory')) # 6 red
                    elif row.target_ds_characteristic or row.target_ds_unique:
                        records.append(r | dict(kind='characteristic')) # 5 orange
                    else:
                        records.append(r | dict(kind='consensus')) # 4 green
    pairs_df = pd.DataFrame.from_records(records)
    pairs_df.to_csv(args.out_file, sep='\t')
