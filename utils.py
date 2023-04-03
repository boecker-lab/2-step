from argparse import Namespace
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sn
# import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import re
from classyfire import get_onehot, get_binary
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Iterable, Callable, Literal
import logging
from time import time
from datetime import timedelta
import json

from features import features

logger = logging.getLogger('rtranknet.utils')
info = logger.info
warning = logger.warning

REL_COLUMNS = ['column.length', 'column.id', 'column.particle.size', 'column.temperature',
               'column.flowrate']
REL_ONEHOT_COLUMNS = ['class.pH.A', 'class.pH.B', 'class.solvent']



def weight_stats(pkl, confl=[]):
    x = pickle.load(open(pkl, 'rb'))
    confl_weights = [_[2] for _ in x if _[0] in confl]
    nonconfl_weights = [_[2] for _ in x if _[0] not in confl]
    print(pd.DataFrame({'nonconfl': nonconfl_weights}).describe())
    print(pd.DataFrame({'confl': confl_weights}).describe())

def rt_diff_weight_fun(rt_diff, weight=1, a=20, b=0.75, max_rt=None, max_rt_ratio=0.05,
                       min_b=0.1, max_b=0.5):
    if (max_rt is not None):
        b = min(max(max_rt_ratio * max_rt, min_b), max_b)
    return (weight              # upper asymptote
            / (1 +
               np.exp(-(
                   a)          # slope
                      * (rt_diff - b))) ** (1))

def pair_weights(smiles1: str, smiles2: str, rt_diff: float,
                 nr_group_pairs: int, nr_group_pairs_max: int,
                 confl_weights_modifier: float, confl_pair_list: Iterable[frozenset]=[],
                 cutoff:float=1e-4, only_confl=False, weight_steepness=20,
                 weight_mid=0.75, max_rt=None) -> Optional[float]:
    # group (~dataset) size balancing modifier
    base_weight = nr_group_pairs_max / nr_group_pairs # roughly between 1 and 500
    # conflicting (-> important) pair modifier
    if (frozenset([smiles1, smiles2]) in confl_pair_list):
        base_weight *= confl_weights_modifier
    elif only_confl:
        base_weight = 0
    # rt diff weight modifier
    base_weight = rt_diff_weight_fun(rt_diff, base_weight, a=weight_steepness, b=weight_mid,
                                     max_rt=max_rt)
    # DEBUG
    # if (random() * 100 < 1):
    #     print(f'weights:\t{rt_diff=:.2f}\t{max_rt=:.2f}\t->{base_weight:.2f}')
    return None if base_weight < cutoff else base_weight



# def plot_fun(weights, fun):
#     x = np.arange(0, 2, 0.001)
#     for w in weights:
#         plt.plot(x, [fun(xi, w) for xi in x], label=f'w={w}')
#         plt.axvline(0.5)
#         plt.axvline(1)
#         print(f'{w=}: \t{fun(0.2, w)=:.5f}\t{fun(0.5, w)=:.5f}\t{fun(0.8, w)=:.5f}')
#     # plt.yscale('log')
#     plt.legend()
#     plt.show()


def get_column_scaling(cols, repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData/',
                       scale_dict={}):
    if (any(c not in scale_dict for c in cols)):
        # load stored info
        if (len(scale_dict) > 0):
            warning(f'scaling: replacing {scale_dict.keys()} with stored values')
        scale_dict.update(json.load(open(os.path.join(repo_root_folder, 'scaling.json'))))
    return (np.array([scale_dict[c]['mean'] for c in cols]),
            np.array([scale_dict[c]['std'] for c in cols]))

def split_arrays(arrays, sizes: tuple, split_info=None, stratify=None):
    for a in arrays:            # all same shape
        assert (len(a) == len(arrays[0])), f'not all arrays to split have the same size, {len(a)} != {len(arrays[0])}'
    # if split info is provided (i.e., whether datapoint should be train/test/val)
    # check whether arrays can be split that way
    if (split_info is not None):
        assert (len(arrays[0]) == len(split_info)), f'split_info (#={len(split_info)}) does not have the same size as the arrays (#={len(arrays[0])})'
        for split, kind in [(sizes[0], 'test'), (sizes[1], 'val')]:
            assert (split == 0 or len([s for s in split_info if s == kind]) > 0), f'not enough {kind} data (required split {split})'
        train_indices = np.argwhere(np.asarray(split_info) == 'train').ravel()
        test_indices = np.argwhere(np.asarray(split_info) == 'test').ravel()
        val_indices = np.argwhere(np.asarray(split_info) == 'val').ravel()
    else:
        indices = np.arange(len(arrays[0]))
        train_indices, test_indices = (train_test_split(indices, test_size=sizes[0],
                                                        stratify=stratify)
                                       if sizes[0] > 0 else (indices, indices[:0]))
        train_indices, val_indices = (train_test_split(train_indices, test_size=sizes[1],
                                                       stratify=np.asarray(stratify)[train_indices])
                                      if sizes[1] > 0 else (train_indices, train_indices[:0]))
    print(f'split {len(arrays[0])} into {len(train_indices)} train data, '
          f'{len(val_indices)} validation data and {len(test_indices)} test data')
    return ([a[train_indices] for a in arrays],
            [a[val_indices] for a in arrays],
            [a[test_indices] for a in arrays],
            (train_indices, val_indices, test_indices))

def reduce_features(values, r_squared_thr=0.96, std_thr=0.01, verbose=True):
    df = pd.DataFrame(values)
    # filter features with low stddev
    filtered = (df.std() > std_thr)
    if verbose:
        print('filtering', filtered[~filtered].index)
    df = df.loc[:, filtered]
    # filter correlated features
    corrs = df.corr()
    corr_vars = [(i, j) for i, j in zip(*np.where(corrs**2 >= r_squared_thr))
                 if i < j and i != j]
    sorted_rels = sorted(
        [(c, {p[0] if p[1] == c else p[1]
              for p in corr_vars if c in p})
         for c in set(c for cp in corr_vars for c in cp)],
        key=lambda x: len(x[1]),
        reverse=True)
    removed_vars = []
    for c, cs in sorted_rels:
        if c not in removed_vars:
            removed_vars.append(c)
    if verbose:
        print('filtering', df.columns[removed_vars])
    df.drop(df.columns[removed_vars], axis=1, inplace=True)
    return df, removed_vars

@dataclass
class Data:
    df: Optional[pd.DataFrame] = None
    use_compound_classes: bool = False
    use_system_information: bool = False
    metadata_void_rt: bool = False
    cache_file: str = 'cached_descs.pkl'
    classes_l_thr: float = 0.005
    classes_u_thr: float = 0.025
    use_usp_codes: bool = False
    custom_features: List[str] = field(default_factory=list)
    use_hsm: bool = False
    use_tanaka: bool = False
    use_newonehot: bool = False
    repo_root_folder: str = '/home/fleming/Documents/Projects/RtPredTrainingData'
    custom_column_fields: Optional[list] = None
    columns_remove_na: bool = True
    hsm_fields: List[str] = field(default_factory=lambda: ['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)'])
    tanaka_fields: List[str] = field(default_factory=lambda: ['kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1'])
    graph_mode: bool = False
    void_info: dict = field(default_factory=dict)
    fallback_column: str = 'Waters ACQUITY UPLC BEH C18' # can be 'average'
    fallback_metadata: str = '0045'                       # can be 'average'
    encoder: Literal['dmpnn', 'dualmpnnplus', 'dualmpnn'] = 'dmpnn'
    graph_args: Optional[Namespace] = None
    sys_scales: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x_features = None
        self.graphs = None
        self.x_classes = None
        self.x_info = None
        self.train_x = None
        self.val_x = None
        self.test_x = None
        self.train_y = None
        self.val_y = None
        self.test_y = None
        self.features_indices = None
        self.classes_indices = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.datasets_df = None
        self.descriptors = None
        self.ph = None


    def compute_graphs(self):
        info('computing graphs')
        t0 = time()
        smiles_unique = set(self.df.smiles)
        if (self.encoder == 'dmpnn'):
            from chemprop.features import mol2graph
            graphs_unique = {s: mol2graph([s]) for s in smiles_unique}
        elif (self.encoder.lower() in ['dualmpnnplus', 'dualmpnn']):
            import sys
            sys.path.append('../CD-MVGNN')
            from dglt.data.featurization.mol2graph import mol2graph
            graph_dict = {}
            graphs_unique = {s: mol2graph([s], graph_dict, self.graph_args)
                             for s in smiles_unique}
        self.graphs = np.array([graphs_unique[s] for s in self.df.smiles])
        info(f'computing graphs done ({str(timedelta(seconds=time() - t0))} elapsed)')
        # self.graphs = np.array([mol2graph([s]) for s in self.df.smiles])

    def compute_features(self,
                         filter_features=None,
                         n_thr=None,
                         recompute=False,
                         mode='rdkit',
                         verbose=False,
                         add_descs=False,
                         add_desc_file='/home/fleming/Documents/Projects/rtranknet/data/qm_merged.csv'):
        if (self.x_features is not None and self.get_y() is not None and not recompute):
            print(
                'features are already computed and `recompute` is not specified, do nothing'
            )
            return
        smiles_unique = list(set(self.df.smiles))
        smiles_pos = [smiles_unique.index(s) for s in self.df.smiles]
        features_unique, self.descriptors = features(smiles_unique, filter_=filter_features, verbose=verbose,
                                                     custom_features=self.custom_features, mode=mode,
                                                     add_descs=add_descs, add_desc_file=add_desc_file)
        self.x_features = features_unique[smiles_pos]
        if (n_thr is not None):
            self.x_features = self.x_features[:, :n_thr]

    def df_classes(self):
        def match_or_nan(id_pattern, field):
            if (not isinstance(field, str) or field.strip() == ''):
                return np.nan
            match = re.search(id_pattern, field)
            return match[0] if match is not None else np.nan

        classyfire_columns = [
            c for c in self.df.columns if c.startswith('classyfire.')
        ]
        if (len(classyfire_columns) == 0):
            raise Exception('no classyfire classes in df!')
        id_pattern = re.compile(r'CHEMONTID:\d+')
        ids = self.df[classyfire_columns].apply(
            lambda row:
            [match_or_nan(id_pattern, field) for field in row],
            axis=1)
        return ids.to_list()

    def compute_classes(self, classes=None, max_rank=None, all_classes=False):
        if (classes is None):
            classes = self.df_classes()
        if all_classes:
            onehots = [[get_onehot(row[i], i) for i in range(
                min((max_rank if max_rank is not None else len(row)), len(row)))]
                       for row in classes]
            self.x_classes = np.array([np.concatenate(row) for row in onehots])
        else:
            self.x_classes = np.array([get_binary(oids, l_thr=self.classes_l_thr, u_thr=self.classes_u_thr)
                                       for oids in classes])

    def compute_system_information(self, onehot_ids=False, other_dataset_ids=None,
                                   use_usp_codes=False, use_hsm=False, use_tanaka=False, use_newonehot=False,
                                   repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData',
                                   custom_column_fields=None, remove_na=True, drop_hsm_dups=False,
                                   fallback_column='Waters ACQUITY UPLC BEH C18', hsm_fallback=True,
                                   col_fields_fallback=True, fallback_metadata='0045',
                                   hsm_fields=['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)'],
                                   tanaka_fields=['kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1']):
        global REL_COLUMNS
        if (onehot_ids):
            if (other_dataset_ids is None):
                self.sorted_dataset_ids = sorted(set(_.split('_')[0] for _ in self.df.id))
            else:
                self.sorted_dataset_ids = other_dataset_ids
            eye = np.eye(len(self.sorted_dataset_ids))
            self.x_info = eye[list(map(self.sorted_dataset_ids.index, (_.split('_')[0] for _ in self.df.id)))]
            return
        fields = []
        names = []
        if (use_hsm):
            hsm = pd.read_csv(os.path.join(repo_root_folder, 'resources/hsm_database/hsm_database.txt'), sep='\t')
            hsm_cf = 'name_new'
            hsm_counts = hsm.value_counts(hsm_cf)
            hsm_dups = hsm_counts.loc[hsm_counts > 1].index.tolist()
            hsm.drop_duplicates([hsm_cf], keep=False if drop_hsm_dups else 'last', inplace=True)
            hsm.set_index(hsm_cf, drop=False, verify_integrity=True, inplace=True)
            self.df.loc[pd.isna(self.df['column.name']), 'column.name'] = 'unknown' # instead of NA
            for c in self.df['column.name'].unique():
                if (c not in hsm[hsm_cf].tolist()):
                    if (not hsm_fallback):
                        raise Exception(
                            f'no HSM data for {", ".join([str(c) for c in set(self.df["column.name"]) if c not in hsm[hsm_cf].tolist()])}')
                    else:
                        if (fallback_column == 'average'):
                            fallback = pd.Series(name=c, dtype='float64')
                            warning(f'using average HSM values for column {c}')
                        elif (fallback_column == 'zeros'):
                            fallback = pd.Series(name=c, dtype='float64')
                            warning(f'using zeros HSM values for column {c}')
                        else:
                            fallback = pd.DataFrame(hsm.loc[fallback_column]).transpose()
                            fallback.index = [c]
                        hsm = pd.concat([hsm, fallback], axis=0)
                elif (c in hsm_dups):
                    warning(f'multiple HSM entries exist for column {c}, the last entry is used')
            means, scales = get_column_scaling(hsm_fields, repo_root_folder=repo_root_folder,
                                               scale_dict=self.sys_scales)
            fields.append((hsm.loc[self.df['column.name'], hsm_fields].astype(float).values - means) / scales)
        if (use_tanaka):
            tanaka = pd.read_csv(os.path.join(repo_root_folder, 'resources/tanaka_database/tanaka_database.txt'), sep='\t')
            tanaka_cf = 'name_new'
            tanaka_counts = tanaka.value_counts(tanaka_cf)
            tanaka_dups = tanaka_counts.loc[tanaka_counts > 1].index.tolist()
            tanaka.drop_duplicates([tanaka_cf], keep=False if drop_hsm_dups else 'last', inplace=True)
            tanaka.set_index(tanaka_cf, drop=False, verify_integrity=True, inplace=True)
            self.df.loc[pd.isna(self.df['column.name']), 'column.name'] = 'unknown' # instead of NA
            for c in self.df['column.name'].unique():
                if (c not in tanaka[tanaka_cf].tolist()):
                    if (not hsm_fallback):
                        raise Exception(
                            f'no Tanaka data for {", ".join([str(c) for c in set(self.df["column.name"]) if c not in tanaka[tanaka_cf].tolist()])}')
                    else:
                        if (fallback_column == 'average'):
                            fallback = pd.Series(name=c, dtype='float64')
                            warning(f'using average Tanaka values for column {c}')
                        elif (fallback_column == 'zeros'):
                            fallback = pd.Series(name=c, dtype='float64')
                            warning(f'using zeros Tanaka values for column {c}')
                        else:
                            fallback = pd.DataFrame(tanaka.loc[fallback_column]).transpose()
                            fallback.index = [c]
                        tanaka = pd.concat([tanaka, fallback], axis=0)
                elif (c in tanaka_dups):
                    warning(f'multiple Tanaka entries exist for column {c}, the last entry is used')
            means, scales = get_column_scaling(tanaka_fields, repo_root_folder=repo_root_folder,
                                               scale_dict=self.sys_scales)
            fields.append((tanaka.loc[self.df['column.name'], tanaka_fields].astype(float).values - means) / scales)
        field_names = custom_column_fields if custom_column_fields is not None else REL_COLUMNS
        na_columns = [col for col in field_names if self.df[col].isna().any()]
        if (len(na_columns) > 0):
            if (col_fields_fallback):
                if (fallback_metadata == 'average' or fallback_metadata == 'zeros'):
                    pass
                else:
                    column_information = pd.read_csv(os.path.join(
                    repo_root_folder, 'processed_data', fallback_metadata, f'{fallback_metadata}_metadata.txt'),
                                                 sep='\t')
                    overwritten_columns = [c for c, all_nans in self.df.loc[
                        self.df[field_names].isna() .any(axis=1), field_names].isna().all().items()
                                           if not all_nans]
                    warning(f'some values if the columns {", ".join(overwritten_columns)} will be overwritten with fallback values!')
                    warning('the following datasets don\'t have all the specified column metadata '
                            f'and will get fallback values: {self.df.loc[self.df[field_names].isna().any(axis=1)].dataset_id.unique().tolist()}')
                    self.df.loc[self.df[field_names].isna().any(axis=1), field_names] = column_information[field_names].iloc[0].tolist()
            elif (remove_na):
                print('removed columns containing NA values: ' + ', '.join(na_columns))
                field_names = [col for col in field_names if col not in na_columns]
            else:
                print('WARNING: system data contains NA values, the option to remove these columns was disabled though! '
                      + ', '.join(na_columns))
        means, scales = get_column_scaling(field_names, repo_root_folder=repo_root_folder,
                                           scale_dict=self.sys_scales)
        fields.append((self.df[field_names].astype(float).values - means) / scales)
        names.extend(field_names)
        if (use_usp_codes):
            codes = ['L1', 'L10', 'L11', 'L43', 'L109']
            codes_vector = (lambda code: np.eye(len(codes))[codes.index(code)]
                            if code in codes else np.zeros(len(codes)))
            code_fields = np.array([codes_vector(c) for c in self.df['column.usp.code']])
            # NOTE: not scaled!
            fields.append(code_fields)
        if (use_newonehot):
            onehot_fields = [c for c in self.df if any(
                c.startswith(prefix + '_') for prefix in REL_ONEHOT_COLUMNS)]
            print('using onehot fields', ', '.join(onehot_fields))
            fields.append(self.df[onehot_fields].astype(float).values)
            # NOTE: not scaled!
        # np.savetxt('/tmp/sys_array.txt', np.concatenate(fields, axis=1), fmt='%.2f')
        self.x_info = np.concatenate(fields, axis=1)
        self.custom_column_fields = names

    def get_y(self):
        return np.array(self.df.rt)

    def get_x(self):
        if (self.x_features is None):
            self.compute_features()
        self.features_indices = [0, self.x_features.shape[1] - 1]
        if (not self.use_compound_classes and not self.use_system_information):
            return self.x_features
        if (self.use_compound_classes and self.x_classes is None):
            self.compute_classes()
        if (self.use_system_information and self.x_info is None):
            self.compute_system_information(use_usp_codes=self.use_usp_codes,
                                            use_hsm=self.use_hsm, use_tanaka=self.use_tanaka,
                                            use_newonehot=self.use_newonehot,
                                            repo_root_folder=self.repo_root_folder,
                                            custom_column_fields=self.custom_column_fields,
                                            remove_na=self.columns_remove_na,
                                            hsm_fields=self.hsm_fields, tanaka_fields=self.tanaka_fields,
                                            fallback_column=self.fallback_column,
                                            fallback_metadata=self.fallback_metadata)
        xs = np.concatenate(list(filter(lambda x: x is not None, (self.x_features, self.x_classes))),
                            axis=1)
        self.classes_indices = ([xs.shape[1] - self.x_classes.shape[1], xs.shape[1] - 1]
                                if self.use_compound_classes else None)
        return (xs, self.x_info)

    def get_graphs(self):
        if (self.graphs is None):
            self.compute_graphs()
        return self.graphs

    def add_dataset_id(self, dataset_id,
                       repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData/',
                       void_rt=0.0, isomeric=True, split_type='train'):
        global REL_ONEHOT_COLUMNS
        paths = [os.path.join(repo_root_folder, 'processed_data', dataset_id,
                              f'{dataset_id}_rtdata_canonical_success.txt'),
                 os.path.join(repo_root_folder, 'processed_data', dataset_id,
                              f'{dataset_id}_rtdata_isomeric_success.txt'),
                 os.path.join(repo_root_folder, 'raw_data', dataset_id,
                              f'{dataset_id}_rtdata.txt')]
        if (not os.path.exists(paths[0])):
            if (os.path.exists(paths[2])):
                warning(f'processed rtdata does not exist for dataset {dataset_id}, '
                        'using raw data')
                df = pd.read_csv(paths[2], sep='\t')
                df.set_index('id', inplace=True, drop=False)
                df.file = paths[2]
                df['smiles'] = df['pubchem.smiles.canonical']
            else:
                raise Exception(f'no data found for dataset {dataset_id}, searched {paths=}')
        else:
            df = pd.read_csv(paths[0], sep='\t')
            df.set_index('id', inplace=True, drop=False)
            if (isomeric):
                if (not os.path.exists(paths[1])):
                    warning(f'--isomeric is set, but no isomeric data can be found for {dataset_id}; '
                            'only canonical data will be used')
                else:
                    df_iso = pd.read_csv(paths[1], sep='\t')
                    df_iso.set_index('id', inplace=True, drop=False)
                    df.update(df_iso)
            df.file = paths[0]
            df['smiles'] = df['smiles.std']
        df['dataset_id'] = df.id.str.split('_', expand=True)[0]
        if self.use_system_information or self.metadata_void_rt:
            column_information = pd.read_csv(os.path.join(
                os.path.dirname(df.file), f'{dataset_id}_metadata.txt'),
                sep='\t')
            column_information['dataset_id'] = [str(x).rjust(4, '0') for x in column_information['id']]
            # NOTE: only set when only one constant pH value is found for all parts of the gradient
            column_information['ph'] = [ph_desc[0] if len(
                ph_desc:=(r[['eluent.A.pH', 'eluent.B.pH', 'eluent.C.pH', 'eluent.D.pH']].replace(0, np.nan).dropna().drop_duplicates()))
                                        == 1 else np.nan for i, r in column_information.iterrows()]
            del column_information['id']
            df = df.merge(column_information, on='dataset_id')
        # rows without RT data are useless
        df = df[~pd.isna(df.rt)]
        # so are compounds (smiles) with multiple rts
        # unless they're the same (TODO: threshold)
        old_len0 = len(df)
        df = df.drop_duplicates(['smiles', 'rt'])
        old_len1 = len(df)
        df = df.drop_duplicates('smiles', keep=False)
        print(f'{dataset_id}: removing duplicate measurements, {old_len0}→{old_len1}→{len(df)}')
        if (self.metadata_void_rt and 'column.t0' in df.columns):
            void_rt = df['column.t0'].iloc[0] * 2 # NOTE: 2 or 3?
        self.void_info[df.dataset_id.iloc[0]] = void_rt
        # flag dataset as train/val/test
        df['split_type'] = split_type
        if (self.df is None):
            self.df = df
        else:
            self.df = pd.concat([self.df, df], ignore_index=True)


    @staticmethod
    def from_raw_file(f, void_rt=0.0, graph_mode=False,
                      metadata_void_rt=False, **extra_data_args):
        # get header
        pot_header = open(f).readlines()[0].strip().split('\t')
        if ('rt' not in pot_header):
            # assume there is no header
            df = pd.read_csv(f, sep='\t', header=None)
            if (len(df.columns) == 3):
                # minimal case
                df.columns = ['inchikey', 'smiles', 'rt']
            else:
                raise NotImplementedError(
                    f'raw file with {len(df.columns)} columns and no header (at least not with rt)')
        else:
            df = pd.read_csv(f, sep='\t')
            if (metadata_void_rt and 'column.t0' in df.columns):
                void_rt = df['column.t0'].iloc[0] * 2
            if ('smiles.std' in df.columns):
                df['smiles'] = df['smiles.std']
        print(f'read raw file {f} with columns {df.columns.tolist()} ({void_rt=})')
        df.file = f
        # rows without RT data are useless
        df = df[~pd.isna(df.rt)]
        # get dataset ID(s) and void time(s)
        if ('dataset_id' not in df.columns):
            # add dummy dataset_id
            df['dataset_id'] = os.path.basename(f)
        if (not metadata_void_rt or 'column.t0' not in df.columns):
            df['column.t0'] = void_rt
        void_info = {t[0]: t[1] for t in set(
            df[['dataset_id', 'column.t0']].itertuples(index=False))}
        return Data(df=df, graph_mode=graph_mode,
                    void_info=void_info,
                    **extra_data_args)

    def balance(self):
        if ('dataset_id' not in self.df.columns):
            raise Exception('cannot balance without Dataset ID')
        g = self.df.groupby('dataset_id')
        df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
        self.heldout = pd.DataFrame(self.df.loc[~self.df.id.isin(df.id)])
        self.df = df

    def features_from_cache(self, cache_file):
        loaded = pickle.load(open(cache_file, 'rb'))
        if (len(loaded) == 3 and isinstance(loaded[0][0], np.ndarray)
                and len(loaded[0][0].shape) > 1):
            ((self.train_x, self.train_y), (self.val_x, self.val_y),
             (self.test_x, self.test_y)) = loaded
        elif (len(loaded) == 2 and isinstance(loaded[0], np.ndarray)):
            self.x_features, self.y = loaded
        else:
            raise Exception('could not load cache!')

    def standardize(self, other_scaler=None):
        if (self.train_x is None):
            raise Exception('feature standardization should only be applied '
                            'after data splitting')
        # standardize data, but only `features`, NaNs can be transformed to 0
        if (self.features_indices[1] - self.features_indices[0] + 1) == 0:
            # no features, don't do anything
            return
        if (other_scaler is None):
            scaler = StandardScaler()
            scaler.fit(self.train_x[:, :self.features_indices[1]+1])
            self.scaler = scaler
        else:
            scaler = other_scaler
        if (len(self.train_x) > 0):
            self.train_x[:, :self.features_indices[1]+1] = np.nan_to_num(scaler.transform(
                self.train_x[:, :self.features_indices[1]+1]))
        if (len(self.val_x) > 0):
            self.val_x[:, :self.features_indices[1]+1] = np.nan_to_num(scaler.transform(
                self.val_x[:, :self.features_indices[1]+1]))
        if (len(self.test_x) > 0):
            self.test_x[:, :self.features_indices[1]+1] = np.nan_to_num(scaler.transform(
                self.test_x[:, :self.features_indices[1]+1]))

    def nan_columns_to_average(self):
        nan_indices = np.where(np.isnan(self.train_x).any(axis=0))[0]
        means = np.nanmean(self.train_x[:, nan_indices], axis=0)
        train_nan_rows = np.isnan(self.train_x).any(axis=1)
        val_nan_rows = np.isnan(self.val_x).any(axis=1)
        test_nan_rows = np.isnan(self.test_x).any(axis=1)
        for mean_, nan_index in zip(means, nan_indices):
            self.train_x[train_nan_rows, nan_index] = mean_
            self.val_x[val_nan_rows, nan_index] = mean_
            self.test_x[test_nan_rows, nan_index] = mean_

    def nan_columns_to_zeros(self):
        nan_indices = np.where(np.isnan(self.train_x).any(axis=0))[0]
        train_nan_rows = np.isnan(self.train_x).any(axis=1)
        val_nan_rows = np.isnan(self.val_x).any(axis=1)
        test_nan_rows = np.isnan(self.test_x).any(axis=1)
        for nan_index in nan_indices:
            self.train_x[train_nan_rows, nan_index] = 0.0
            self.val_x[val_nan_rows, nan_index] = 0.0
            self.test_x[test_nan_rows, nan_index] = 0.0


    def reduce_f(self, r_squared_thr=0.96, std_thr=0.01, verbose=True):
        if (self.train_x is None):
            raise Exception('feature reduction should only be applied '
                            'after data splitting')
        # remove unnecessary features
        train_x_new, removed = reduce_features(self.train_x,
                                               r_squared_thr=r_squared_thr,
                                               std_thr=std_thr,
                                               verbose=verbose)
        self.train_x = np.delete(self.train_x, removed, axis=1)
        self.val_x = np.delete(self.val_x, removed, axis=1)
        self.test_x = np.delete(self.test_x, removed, axis=1)

    def split_data(self, split=(0.2, 0.05)):
        if ('split_type' in self.df.columns and self.df.split_type.nunique() > 1):
            split_info = self.df.split_type
        else:
            split_info = None
        if (self.graph_mode):
            ((self.train_graphs, self.train_x, self.train_sys, self.train_y),
             (self.val_graphs, self.val_x, self.val_sys, self.val_y),
             (self.test_graphs, self.test_x, self.test_sys, self.test_y,),
             (self.train_indices, self.val_indices, self.test_indices)) = split_arrays(
                 (self.get_graphs(), *self.get_x(), self.get_y()), split,
                 split_info=split_info, stratify=self.df.dataset_id.tolist())
        else:
            ((self.train_x, self.train_y),
             (self.val_x, self.val_y),
             (self.test_x, self.test_y),
             (self.train_indices, self.val_indices, self.test_indices)) = split_arrays(
                 (self.get_x(), self.get_y()), split,
                 split_info=split_info, stratify=self.df.dataset_id.tolist())
            self.train_graphs = self.val_graphs = self.test_graphs = None

    def get_raw_data(self):
        if (self.graph_mode):
            return self.get_graphs(), self.get_x(), self.get_y()
        else:
            return self.get_x(), self.get_y()

    def get_split_data(self, split=(0.2, 0.05)):
        if ((any(d is None for d in [
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y
        ] + ([self.train_graphs, self.val_graphs, self.test_graphs] if self.graph_mode else [])))):
            self.split_data(split)
        return ((self.train_graphs, self.train_x, self.train_sys, self.train_y),
                (self.val_graphs, self.val_x, self.val_sys, self.val_y),
                (self.test_graphs, self.test_x, self.test_sys, self.test_y))

def export_predictions(data, preds, out, mode='all'):
    if (mode == 'all'):
        df = pd.DataFrame(data.df.iloc[np.concatenate((data.train_indices, data.test_indices, data.val_indices))])
    elif (mode == 'test'):
        df = pd.DataFrame(data.df.iloc[data.test_indices])
    else:
        raise NotImplementedError(mode)
    df['roi'] = preds
    df[['smiles', 'rt', 'roi']].to_csv(out, sep='\t', index=False, header=False)

def naive_void_est(df, perc_mean=1):
    sorted_df = df.sort_values(by='rt')
    x = sorted_df.rt.values - np.concatenate([sorted_df.rt.values[:1], sorted_df.rt.values])[:-1]
    i = max(0, (x < (np.mean(x) * perc_mean)).argmin(0))
    return sorted_df.rt.iloc[i]
