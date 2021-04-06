import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sn
# import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from scipy.sparse import  issparse
import pickle
import os
import re
from classyfire import get_onehot, get_binary
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from logging import warning, info
import sys

from features import features


REL_COLUMNS = ['column.length', 'column.id', 'column.particle.size', 'column.temperature',
               'column.flowrate', 'eluent.A.h2o', 'eluent.A.meoh', 'eluent.A.acn',
               'eluent.A.formic', 'eluent.A.nh4ac', 'eluent.A.nh4form',
               'eluent.B.h2o', 'eluent.B.meoh', 'eluent.B.acn', 'eluent.B.formic',
               'eluent.B.nh4ac', 'eluent.B.nh4form', 'gradient.start.A',
               'gradient.end.A']


def csr2tf(csr):
    indices = []
    values = []
    for (i, j), v in csr.todok().items():
        indices.append([i, j])
        values.append(v)
    return tf.sparse.SparseTensor(indices, values, csr.shape)


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, class_weights=None, batch_size=32, shuffle=True, delta=1,
                 pair_step=1, pair_stop=None, use_weights=True,
                 weight_steep=4, weight_mid=0.75, void=None,
                 y_neg=False, multix=False):
        self.x = x
        self.y = y
        self.class_weights = class_weights
        self.delta = delta
        self.multix = multix
        self.use_weights = use_weights
        self.weight_steep = weight_steep
        self.weight_mid = weight_mid
        self.pair_step = pair_step
        self.pair_stop = pair_stop
        self.void = void
        self.y_neg = y_neg
        self.x1_indices, self.x2_indices, self.y_trans, self.weights = self._transform_pairwise(
            x, y, class_weights)
        if (shuffle):
            perm = np.random.permutation(self.y_trans.shape[0])
            self.x1_indices = self.x1_indices[perm]
            self.x2_indices = self.x2_indices[perm]
            self.y_trans = self.y_trans[perm]
            self.weights = self.weights[perm]
        self.batch_size = batch_size

    @staticmethod
    def weight_fn(x, steep=4, mid=0.75):
        """sigmoid function with f(0) → 0, f(2) → 1, f(0.75) = 0.5"""
        return 1 / (1 + np.exp(-steep * (x - mid)))

    def _transform_pairwise(self, x, y, class_weights):
        x1_indices = []
        x2_indices = []
        y_trans = []
        weights = []
        for i in range(len(y)):
            for j in range(i + 1, (len(y) if self.pair_stop is None else
                                   min(i + self.pair_stop, len(y))),
                           self.pair_step):
                pos_idx, neg_idx = (i, j) if y[i] > y[j] else (j, i)
                # void
                if (self.void is not None and y[i] < self.void
                    and y[j] < self.void):
                    # don't take pairs where both compounds are in void volume
                    continue
                # balanced class
                if 1 != (-1)**(pos_idx + neg_idx):
                    x1_indices.append(pos_idx)
                    x2_indices.append(neg_idx)
                    y_trans.append(1)
                else:
                    x1_indices.append(neg_idx)
                    x2_indices.append(pos_idx)
                    y_trans.append(-1 if self.y_neg else 0)
                # weights
                c_weight = (min(class_weights[i], class_weights[j])
                            if class_weights is not None else 1)
                weights.append(c_weight * self.weight_fn(
                    y[pos_idx] - y[neg_idx], self.weight_steep, self.weight_mid)
                               if self.use_weights else 1)
        return np.asarray(x1_indices), np.asarray(x2_indices), np.asarray(
            y_trans), np.asarray(weights)

    def __len__(self):
        return np.ceil(self.y_trans.shape[0] / self.batch_size).astype(int)

    def __getitem__(self, index):
        i = index * self.batch_size
        if (self.multix):
            X1_trans = [xi[self.x1_indices[i:(i + self.batch_size)]] for xi in self.x]
            X2_trans = [xi[self.x2_indices[i:(i + self.batch_size)]] for xi in self.x]
        else:
            X1_trans = self.x[self.x1_indices[i:(i + self.batch_size)]]
            X2_trans = self.x[self.x2_indices[i:(i + self.batch_size)]]
        if (issparse(X1_trans)):
            # convert to sparse TF tensor
            X1_trans = csr2tf(X1_trans)
            X2_trans = csr2tf(X2_trans)
        return [X1_trans,
                X2_trans], self.y_trans[i:(i + self.batch_size)], self.weights[i:(i + self.batch_size)]

    def get_df(self, x_desc='features'):
        return pd.DataFrame({x_desc: self.x, 'rt': self.y})

def get_column_scaling(cols, repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData/'):
    if (not hasattr(get_column_scaling, '_data')):
        sys.path.append(repo_root_folder)
        from pandas_dfs import get_dataset_df
        ds = get_dataset_df()
        info_columns = [c for c in ds.columns
                        if re.match(r'^(column|gradient|eluent)\..*', c)
                        and 'name' not in c and 'usp.code' not in c]
        # empirical
        s = StandardScaler()
        s.fit(ds[info_columns])
        get_column_scaling._data = {col: {'mean': mean, 'std': scale}
                                    for col, mean, scale
                                    in zip(info_columns, s.mean_, s.scale_)}
        # manual
        get_column_scaling._data.update({col: {'mean': 50., 'std': 50.} # values 0-100
                                         for col in info_columns
                                         if (col.startswith('eluent.')
                                             or col.startswith('gradient.'))})
    return (np.array([get_column_scaling._data[c]['mean'] for c in cols]),
            np.array([get_column_scaling._data[c]['std'] for c in cols]))

def split_arrays(arrays, sizes: tuple, split_info=None):
    for a in arrays:            # all same shape
        assert len(a) == len(arrays[0])
    # if split info is provided (i.e., whether datapoint should be train/test/val)
    # check whether arrays can be split that way
    if (split_info is not None):
        assert len(arrays[0]) == len(split_info)
        for split, kind in [(sizes[0], 'test'), (sizes[1], 'val')]:
            assert split == 0 or len([s for s in split_info if s == kind]) > 0
        train_indices = np.argwhere(np.asarray(split_info) == 'train').ravel()
        test_indices = np.argwhere(np.asarray(split_info) == 'test').ravel()
        val_indices = np.argwhere(np.asarray(split_info) == 'val').ravel()
    else:
        indices = np.arange(len(arrays[0]))
        train_indices, test_indices = (train_test_split(indices, test_size=sizes[0])
                                       if sizes[0] > 0 else (indices, indices[:0]))
        train_indices, val_indices = (train_test_split(train_indices, test_size=sizes[1])
                                      if sizes[1] > 0 else (train_indices, train_indices[:0]))
    print(f'split {len(arrays)} into {len(train_indices)} train data, '
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
    repo_root_folder: str = '/home/fleming/Documents/Projects/RtPredTrainingData'
    custom_column_fields: Optional[list] = None
    columns_remove_na: bool = True
    hsm_fields: List[str] = field(default_factory=lambda: ['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)'])
    graph_mode: bool = False

    def __post_init__(self):
        self.x_features = None
        self.graphs = None
        self.cweights = None
        self.x_classes = None
        self.x_info = None
        self.train_x = None
        self.val_x = None
        self.test_x = None
        self.train_y = None
        self.val_y = None
        self.test_y = None
        self.features_indices = None
        self.info_indices = None
        self.classes_indices = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.datasets_df = None
        self.descriptors = None


    def compute_graphs(self):
        # TODO: compute only unique smiles + multithreaded
        from chemprop.features import mol2graph
        self.graphs = np.array([mol2graph([s]) for s in self.df.smiles])

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
                                   use_usp_codes=False, use_hsm=False,
                                   repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData',
                                   custom_column_fields=None, remove_na=True, drop_hsm_dups=True,
                                   hsm_fields=['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)']):
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
            if (drop_hsm_dups):
                hsm.drop_duplicates(['normalized notation'], keep=False, inplace=True)
            hsm.set_index('normalized notation', drop=False, verify_integrity=drop_hsm_dups, inplace=True)
            if (any(c not in hsm['normalized notation'].tolist() for c in self.df['column.name'])):
                raise Exception(
                    f'no HSM data for {", ".join([str(c) for c in set(self.df["column.name"]) if c not in hsm["normalized notation"].tolist()])}')
            # NOTE: not scaled!
            fields.append(hsm.loc[self.df['column.name'], hsm_fields].values)
        if (custom_column_fields is not None):
            na_columns = [col for col in custom_column_fields if self.df[col].isna().any()]
            if (remove_na):
                print('removed columns containing NA values: ' + ', '.join(na_columns))
                custom_column_fields = [col for col in custom_column_fields if col not in na_columns]
            elif (len(na_columns) > 0):
                print('WARNING: system data contains NA values, the option to remove these columns was disabled though! '
                      + ', '.join(na_columns))
            means, scales = get_column_scaling(custom_column_fields, repo_root_folder=repo_root_folder)
            fields.append((self.df[custom_column_fields].values - means) / scales)
            names.extend(custom_column_fields)
        else:
            na_columns = [col for col in REL_COLUMNS if self.df[col].isna().any()]
            if (len(na_columns) > 0):
                if (remove_na):
                    print('removed columns containing NA values: ' + ', '.join(na_columns))
                    REL_COLUMNS = [col for col in REL_COLUMNS if col not in na_columns]
                else:
                    print('WARNING: system data contains NA values, the option to remove these columns was disabled though! '
                          + ', '.join(na_columns))
            means, scales = get_column_scaling(REL_COLUMNS, repo_root_folder=repo_root_folder)
            fields.append((self.df[REL_COLUMNS].values - means) / scales)
            names.extend(REL_COLUMNS)
        if (use_usp_codes):
            codes = ['L1', 'L10', 'L11', 'L43', 'L109']
            codes_vector = (lambda code: np.eye(len(codes))[codes.index(code)]
                            if code in codes else np.zeros(len(codes)))
            code_fields = np.array([codes_vector(c) for c in self.df['column.usp.code']])
            # NOTE: not scaled!
            fields.append(code_fields)
        np.savetxt('/tmp/sys_array.txt', np.concatenate(fields, axis=1), fmt='%.2f')
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
                                            use_hsm=self.use_hsm,
                                            repo_root_folder=self.repo_root_folder,
                                            custom_column_fields=self.custom_column_fields,
                                            remove_na=self.columns_remove_na,
                                            hsm_fields=self.hsm_fields)
        xs = np.concatenate(list(filter(lambda x: x is not None, (self.x_features, self.x_info, self.x_classes))),
                            axis=1)
        self.info_indices = ([self.features_indices[-1] + 1,
                                self.features_indices[-1] + self.x_info.shape[1]]
                                if self.use_system_information else None)
        self.classes_indices = ([xs.shape[1] - self.x_classes.shape[1], xs.shape[1] - 1]
                             if self.use_compound_classes else None)
        print(f'{np.diff(self.features_indices) + 1} molecule features, '
              f'{(np.diff(self.info_indices) + 1) if self.info_indices is not None else 0} column features, '
              f'{(np.diff(self.classes_indices) + 1) if self.classes_indices is not None else 0} molecule class features')
        return xs

    def get_graphs(self):
        if (self.graphs is None):
            self.compute_graphs()
        return self.graphs

    def get_cweights(self):
        return class_weight.compute_sample_weight('balanced', y=self.df.dataset_id)

    def add_dataset_id(self, dataset_id,
                       repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData/',
                       void_rt=0.0, isomeric=True, split_type='train'):
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
            del column_information['id']
            df = df.merge(column_information, on='dataset_id')
        # rows without RT data are useless
        df = df[~pd.isna(df.rt)]
        # filter rows below void RT threshold
        if (self.metadata_void_rt and 'column.t0' in df.columns):
            void_rt = df['column.t0'].iloc[0] * 2 # NOTE: 2 or 3?
        df = df.loc[~(df.rt < void_rt)]
        # flag dataset as train/val/test
        df['split_type'] = split_type
        if (self.df is None):
            self.df = df
        else:
            self.df = self.df.append(df, ignore_index=True)


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
        # filter rows below void RT threshold
        df = df.loc[~(df.rt < void_rt)]
        # add dummy dataset_id
        df['dataset_id'] = os.path.basename(f)
        return Data(df=df, graph_mode=graph_mode, **extra_data_args)

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
            ((self.train_graphs, self.train_x, self.train_y, self.train_cweights),
             (self.val_graphs, self.val_x, self.val_y, self.val_cweights),
             (self.test_graphs, self.test_x, self.test_y, self.test_cweights),
             (self.train_indices, self.val_indices, self.test_indices)) = split_arrays(
                 (self.get_graphs(), self.get_x(), self.get_y(), self.get_cweights()), split,
                 split_info=split_info)
        else:
            ((self.train_x, self.train_y, self.train_cweights),
             (self.val_x, self.val_y, self.val_cweights),
             (self.test_x, self.test_y, self.test_cweights),
             (self.train_indices, self.val_indices, self.test_indices)) = split_arrays(
                 (self.get_x(), self.get_y(), self.get_cweights()), split,
                 split_info=split_info)
            self.train_graphs = self.val_graphs = self.test_graphs = None

    def get_raw_data(self):
        if (self.graph_mode):
            return self.get_graphs(), self.get_x(), self.get_y()
        else:
            return self.get_x(), self.get_y()

    def get_split_data(self, split=(0.2, 0.05)):
        if ((any(d is None for d in [
                self.train_x, self.train_y, self.train_cweights, self.val_x, self.val_y,
                self.val_cweights, self.test_x, self.test_y, self.test_cweights
        ] + ([self.train_graphs, self.val_graphs, self.test_graphs] if self.graph_mode else [])))):
            self.split_data(split)
        return ((self.train_graphs, self.train_x, self.train_y, self.train_cweights),
                (self.val_graphs, self.val_x, self.val_y, self.val_cweights),
                (self.test_graphs, self.test_x, self.test_y, self.test_cweights))

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
