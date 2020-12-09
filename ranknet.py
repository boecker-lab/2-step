import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sn
# import lightgbm as lgb
import tensorflow as tf
from keras import backend as K
from LambdaRankNN import RankNetNN
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from mordred import Calculator, descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, issparse
import multiprocessing as mp
from itertools import combinations
import pickle
import os
from os.path import splitext, basename
import argparse
import re
from classyfire import get_onehot, get_binary
from pprint import pprint
import contextlib
# from directranker.DirectRanker import directRanker


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32, shuffle=True, delta=1,
                 pair_step=1, pair_stop=None, use_weights=True):
        self.x = x
        self.y = y
        self.delta = delta
        self.use_weights = use_weights
        self.pair_step = pair_step
        self.pair_stop = pair_stop
        self.x1_indices, self.x2_indices, self.y_trans, self.weights = self._transform_pairwise(
            x, y)
        if (shuffle):
            perm = np.random.permutation(self.y_trans.shape[0])
            self.x1_indices = self.x1_indices[perm]
            self.x2_indices = self.x2_indices[perm]
            self.y_trans = self.y_trans[perm]
            self.weights = self.weights[perm]
        self.batch_size = batch_size

    @staticmethod
    def weight_fn(x):
        """sigmoid function appr. 0 at 0 and 1 at 2, 0.5 as 0.75"""
        return 1 / (1 + np.exp(-4 * (x - 0.75)))

    def _transform_pairwise(self, x, y):
        x1_indices = []
        x2_indices = []
        y_trans = []
        weights = []
        for i in range(len(y)):
            for j in range(i + 1, (len(y) if self.pair_stop is None else
                                   min(i + self.pair_stop, len(y))),
                           self.pair_step):
                # if (np.abs(self.y[i] - self.y[j]) <= self.delta):
                #     continue
                pos_idx, neg_idx = (i, j) if y[i] > y[j] else (j, i)
                # balanced class
                if 1 != (-1)**(pos_idx + neg_idx):
                    x1_indices.append(pos_idx)
                    x2_indices.append(neg_idx)
                    y_trans.append(1)
                else:
                    x1_indices.append(neg_idx)
                    x2_indices.append(pos_idx)
                    y_trans.append(0)
                weights.append(self.weight_fn(y[pos_idx] - y[neg_idx])
                               if self.use_weights else 1)
        return np.asarray(x1_indices), np.asarray(x2_indices), np.asarray(
            y_trans), np.asarray(weights)

    def __len__(self):
        return np.ceil(self.y_trans.shape[0] / self.batch_size).astype(int)

    def __getitem__(self, index):
        i = index * self.batch_size
        X1_trans = self.x[self.x1_indices[i:(i + self.batch_size)]]
        X2_trans = self.x[self.x2_indices[i:(i + self.batch_size)]]
        if (issparse(X1_trans)):
            # convert to sparse TF tensor
            X1_trans = csr2tf(X1_trans)
            X2_trans = csr2tf(X2_trans)
        # weights = np.asarray(
        #     [1] * X1_trans.shape[0]
        # )  # possibly prevents getting stuck at *some* local minima
        return [X1_trans,
                X2_trans], self.y_trans[i:(i + self.batch_size)], self.weights[i:(i + self.batch_size)]


def csr2tf(csr):
    indices = []
    values = []
    for (i, j), v in csr.todok().items():
        indices.append([i, j])
        values.append(v)
    return tf.sparse.SparseTensor(indices, values, csr.shape)


# m = Chem.MolFromSmiles('C1C(C(OC2=CC(=CC(=C21)O)O)C3=CC(=C(C=C3)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O')
# Chem.MolToInchi(m)
# GraphDescriptors.Chi4v(m)

# Descriptors.descList

# smrt_all = pd.read_csv('/mnt/Data/Projects/RTPred/SMRT/SMRT_dataset.csv')

# m = Chem.MolFromSmiles(test_data.iloc[0, 1])
# Chem.MolFromI
# descc = {g: f(m) for g, f in Descriptors.descList}


def get_morgan_fps(m, r):
    return AllChem.GetMorganFingerprint(m, r)

def get_descriptors():
    features = []
    features.extend([(name, fun, 'rdk') for name, fun in Descriptors.descList])
    features.extend([(name, fun, '3d') for name, fun in
                     [('Asphericity', Descriptors3D.Asphericity),
                      ('Eccentricity', Descriptors3D.Eccentricity),
                      ('InertialShapeFactor', Descriptors3D.InertialShapeFactor),
                      ('NPR1', Descriptors3D.NPR1), ('NPR2', Descriptors3D.NPR2),
                      ('PMI1', Descriptors3D.PMI1), ('PMI2', Descriptors3D.PMI2), ('PMI3', Descriptors3D.PMI3),
                      ('RadiusOfGyration', Descriptors3D.RadiusOfGyration),
                      ('SpherocityIndex', Descriptors3D.SpherocityIndex)]])
    return features

def features(smiles, filter_='rdk', overwrite_cache=False, verbose=False,
             custom_features=[]):
    assert (len(smiles) == len(set(smiles))), 'smiles have to be unique'
    if (not hasattr(features, 'cached')):
        features.cached = {}
    descriptors = get_descriptors()
    if (filter_ is not None):
        filter_fun = {'rdk': lambda t: t[2] == 'rdk',
                      '3d': lambda t: t[2] == '3d',
                      }[filter_]
        descriptors = list(filter(filter_fun, descriptors))
    if (len(custom_features) > 0):
        descriptors = sorted(list(filter(lambda t: t[0] in custom_features, descriptors)),
                          key=lambda t: custom_features.index(t[0]))
    features.descriptors = {name: fun for name, fun, _ in descriptors}
    to_calc = {}
    for s in smiles:
        for fname, ffun, _ in descriptors:
            if ((s, fname) not in features.cached
                or overwrite_cache):
                to_calc.setdefault(s, []).append(fname)
    to_calc = [(smile, descriptors) for smile, descriptors in to_calc.items()]
    if (len(to_calc) > 0):
        features.write_cache = True # cache has to be written in the end
        pool = mp.Pool(mp.cpu_count())
        res = pool.starmap(compute_descriptors, to_calc)
        pool.close()
        res_new = []
        for descs, values, failed in res:
            if (len(failed) > 0 and verbose):
                print('failed', failed)
            res_new.append([(d, v) for d, v in zip(descs, values)])
        features.cached.update({(smile[0], desc): value for smile, smile_res in zip(to_calc, res_new)
                                for desc, value in smile_res})
    return np.array([[features.cached[(smile, desc[0])] for desc in descriptors]
                     for smile in smiles])

def compute_descriptors(smile, descriptors):
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        AllChem.EmbedMolecule(mol)
    except:
        return [descriptors, [np.nan for d in descriptors], descriptors]
    values = []
    failed = []
    for name in descriptors:
        fun = features.descriptors[name]
        try:
            val = fun(mol)
        except:
            val = np.nan
            failed.append(name)
        values.append(val)
    return [descriptors, values, failed]


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


# df, removed = reduce_features(features(test_data.smiles[:100]))


def split_df(df, sizes: tuple):
    train_indices, test_indices = train_test_split(list(range(len(df))),
                                                   test_size=sizes[0])
    train_indices, val_indices = train_test_split(train_indices,
                                                  test_size=sizes[1])
    return df.loc[train_indices], df.loc[val_indices], df.loc[test_indices]


def split_arrays(x, y, sizes: tuple):
    assert x.shape[0] == len(y)
    (train_x, test_x, train_y, test_y,
     train_indices, test_indices) = train_test_split(x,
                                                     y,
                                                     np.arange(x.shape[0]),
                                                     test_size=sizes[0])
    train_x, val_x, train_y, val_y, train_indices, val_indices = train_test_split(train_x,
                                                                      train_y,
                                                                      train_indices,
                                                                      test_size=sizes[1])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (
        train_indices, val_indices, test_indices)


def get_data(df, kind='rdk'):
    return features(df.smiles, kind), (np.array(df.rt) * 10).astype(int)


def eval_(y, preds, epsilon=1):
    assert len(y) == len(preds)
    if (not any(preds)):
        return 0.0
    preds, y = zip(*sorted(zip(preds, y)))
    matches = 0
    total = 0
    for i, j in combinations(range(len(y)), 2):
        diff = y[i] - y[j]
        if (diff < epsilon):
            matches += 1
        total += 1
    return matches / total if not total == 0 else np.nan

def get_column_scaling(cols):
    if (not hasattr(get_column_scaling, '_data')):
        ds = pd.read_csv('/home/fleming/Documents/Projects/rtdata_exploration/data/dataset_info_all.tsv',
                         sep='\t')
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


class Data:
    def __init__(self, df=None, use_compound_classes=False,
                 use_system_information=False, cache_file='cached_descs.pkl',
                 classes_l_thr=0.005, classes_u_thr=0.025, use_usp_codes=False,
                 custom_features=[], use_hsm=False,
                 hsm_data='/home/fleming/Documents/Projects/RtPredTrainingData/hsm.tsv',
                 custom_column_fields=None, columns_remove_na=True):
        self.df = df
        self.x_features = None
        self.x_classes = None
        self.x_info = None
        self.train_x = None
        self.val_x = None
        self.test_x = None
        self.train_y = None
        self.val_y = None
        self.test_y = None
        self.use_compound_classes = use_compound_classes
        self.use_system_information = use_system_information
        self.features_indices = None
        self.info_indices = None
        self.classes_indices = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.cache_file = None
        self.classes_l_thr = classes_l_thr
        self.classes_u_thr = classes_u_thr
        self.datasets_df = None
        self.use_usp_codes = use_usp_codes
        self.custom_features = custom_features
        self.use_hsm = use_hsm
        self.hsm_data = hsm_data
        self.custom_column_fields = custom_column_fields
        self.columns_remove_na = columns_remove_na

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
                                            hsm_data=self.hsm_data,
                                            custom_column_fields=self.custom_column_fields,
                                            remove_na=self.columns_remove_na)
        xs = np.concatenate(list(filter(lambda x: x is not None, (self.x_features, self.x_info, self.x_classes))),
                            axis=1)
        self.info_indices = ([self.features_indices[-1] + 1,
                                self.features_indices[-1] + self.x_info.shape[1]]
                                if self.use_system_information else None)
        self.classes_indices = ([xs.shape[1] - self.x_classes.shape[1], xs.shape[1] - 1]
                             if self.use_compound_classes else None)
        return xs

    def add_dataset_id(self, dataset_id,
                       repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData/',
                       void_rt=0.0, isomeric=True):
        f = os.path.join(repo_root_folder, 'processed_data', dataset_id,
                         f'{dataset_id}_rtdata_canonical_success.txt')
        df = pd.read_csv(f, sep='\t')
        df.set_index('id', inplace=True, drop=False)
        if (isomeric):
            f_iso = os.path.join(repo_root_folder, 'processed_data', dataset_id,
                             f'{dataset_id}_rtdata_isomeric_success.txt')
            df_iso = pd.read_csv(f_iso, sep='\t')
            df_iso.set_index('id', inplace=True, drop=False)
            df.update(df_iso)
        df.file = f
        df['dataset_id'] = df.id.str.split('_', expand=True)[0]
        if self.use_system_information:
            # only numeric values from metadata
            column_information = pd.read_csv(os.path.join(
                repo_root_folder, 'processed_data', dataset_id,
                f'{dataset_id}_metadata.txt'),
                sep='\t')
            column_information['dataset_id'] = [str(x).rjust(4, '0') for x in column_information['id']]
            del column_information['id']
            df = df.merge(column_information, on='dataset_id')
            # if (self.datasets_df is None):
            #     self.datasets_df = pd.read_csv(
            #         os.path.join(repo_root_folder, 'raw_data', 'studies.txt'), sep='\t')
            # df = df.join(pd.concat())
        # rows without RT data are useless
        df = df[~pd.isna(df.rt)]
        # filter rows below void RT threshold
        df = df.loc[~(df.rt < void_rt)]
        if (self.df is None):
            self.df = df
        else:
            self.df = self.df.append(df, ignore_index=True)
        self.df['smiles'] = self.df['smiles.std']

    @staticmethod
    def from_raw_file(f, header=None, void_rt=0.0):
        df = pd.read_csv(f, sep='\t', header=header)
        df.file = f
        if (header is None):
            if (len(df.columns) == 3):
                # minimal case
                df.columns = ['inchikey', 'smiles', 'rt']
            else:
                raise NotImplementedError(
                    f'raw file with {len(df.columns)} columns')
        # rows without RT data are useless
        df = df[~pd.isna(df.rt)]
        # filter rows below void RT threshold
        df = df.loc[~(df.rt < void_rt)]
        return Data(df=df)

    def balance(self):
        if ('dataset_id' not in self.df.columns):
            raise Exception('cannot balance without Dataset ID')
        g = self.df.groupby('dataset_id')
        df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
        self.heldout = pd.DataFrame(self.df.loc[~data.df.id.isin(df.id)])
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

    def compute_features(self,
                         filter_features=None,
                         n_thr=None,
                         recompute=False,
                         verbose=False):
        if (self.x_features is not None and self.get_y() is not None and not recompute):
            print(
                'features are already computed and `recompute` is not specified, do nothing'
            )
            return
        smiles_unique = list(set(self.df.smiles))
        smiles_pos = [smiles_unique.index(s) for s in self.df.smiles]
        features_unique = features(smiles_unique, filter_=filter_features, verbose=verbose,
                                   custom_features=self.custom_features)
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
                                   hsm_data='/home/fleming/Documents/Projects/RtPredTrainingData/hsm.tsv',
                                   custom_column_fields=None, remove_na=True):
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
            hsm = pd.read_csv(hsm_data, sep='\t', index_col=0)
            cols = ['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)']
            if (any(c not in hsm.index for c in self.df['column.name'])):
                raise Exception(
                    f'no HSM data for {", ".join([str(c) for c in set(self.df["column.name"]) if c not in hsm.index])}')
            # NOTE: not scaled!
            fields.append(hsm.loc[self.df['column.name'], cols].values)
        if (custom_column_fields is not None):
            na_columns = [col for col in custom_column_fields if self.df[col].isna().any()]
            if (remove_na):
                print('removed columns containing NA values: ' + ', '.join(na_columns))
                custom_column_fields = [col for col in custom_column_fields if col not in na_columns]
            elif (len(na_columns) > 0):
                print('WARNING: system data contains NA values, the option to remove these columns was disabled though! '
                      + ', '.join(na_columns))
            means, scales = get_column_scaling(custom_column_fields)
            fields.append((self.df[custom_column_fields].values - means) / scales)
            names.extend(custom_column_fields)
        else:
            rel_columns = ['column.length', 'column.id', 'column.particle.size', 'column.temperature',
                           'column.flowrate', 'eluent.A.h2o', 'eluent.A.meoh', 'eluent.A.acn',
                           'eluent.A.formic', 'eluent.A.nh4ac', 'eluent.A.nh4form',
                           'eluent.B.h2o', 'eluent.B.meoh', 'eluent.B.acn', 'eluent.B.formic',
                           'eluent.B.nh4ac', 'eluent.B.nh4form', 'gradient.start.A',
                           'gradient.end.A']
            na_columns = [col for col in rel_columns if self.df[col].isna().any()]
            if (len(na_columns) > 0):
                if (remove_na):
                    print('removed columns containing NA values: ' + ', '.join(na_columns))
                    rel_columns = [col for col in rel_columns if col not in na_columns]
                else:
                    print('WARNING: system data contains NA values, the option to remove these columns was disabled though! '
                          + ', '.join(na_columns))
            means, scales = get_column_scaling(rel_columns)
            fields.append((self.df[rel_columns].values - means) / scales)
            names.extend(rel_columns)
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

    def standardize(self, other_scaler=None):
        if (self.train_x is None):
            raise Exception('feature standardization should only be applied '
                            'after data splitting')
        # standardize data, but only `features`, NaNs can be transformed to 0
        if (other_scaler is None):
            scaler = StandardScaler()
            scaler.fit(self.train_x[:, :self.features_indices[1]+1])
            self.scaler = scaler
        else:
            scaler = other_scaler
        self.train_x[:, :self.features_indices[1]+1] = np.nan_to_num(scaler.transform(
            self.train_x[:, :self.features_indices[1]+1]))
        self.val_x[:, :self.features_indices[1]+1] = np.nan_to_num(scaler.transform(
            self.val_x[:, :self.features_indices[1]+1]))
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
        ((self.train_x, self.train_y), (self.val_x, self.val_y),
         (self.test_x, self.test_y),
         (self.train_indices, self.val_indices, self.test_indices)) = split_arrays(
             self.get_x(), self.get_y(), split)

    def get_raw_data(self):
        return self.get_x(), self.get_y()

    def get_split_data(self, split=(0.2, 0.05)):
        if ((any(d is None for d in [
                self.train_x, self.train_y, self.val_x, self.val_y,
                self.test_x, self.test_y
        ]))):
            self.split_data(split)
        return ((self.train_x, self.train_y), (self.val_x, self.val_y),
                (self.test_x, self.test_y))


def data_test():
    d = Data(use_compound_classes=True, use_system_information=True,
             classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr)
    d.add_dataset_id('0163')
    d.compute_features()
    d.compute_classes(max_rank=4)
    d.split_data()
    d.standardize()

    e = Data.from_raw_file('test.csv')
    # e.compute_features(write_cache=True)
    e.features_from_cache('test_rdk.pkl')
    e.split_data()
    e.standardize()
    e.reduce_f()
    e.get_split_data()

    f = Data(use_compound_classes=False, use_system_information=True)
    cortecs_t3_sets = ['0083', '0084', '0085', '0086', '0087', '0088',
                       '0089', '0090', '0091', '0092', '0093', '0094',
                       '0095', '0096', '0097', '0098', '0099', '0100',
                       '0101', '0140', '0141', '0142', '0143', '0144',
                       '0145', '0146', '0147', '0148', '0149', '0150',
                       '0151', '0152', '0153', '0154', '0155', '0156',
                       '0157', '0158']
    for ds in cortecs_t3_sets:
        f.add_dataset_id(ds, 'canonical')
    f.compute_features()
    # f.compute_classes()
    f.compute_system_information()
    f.split_data()
    f.standardize()



def load_example_data(data_file='test.csv',
                      features_type='rdk',
                      standardize=False,
                      reduce_f=False,
                      n_features=None,
                      cached=True,
                      write_cache=False):
    cache_file = f'{splitext(basename(data_file))[0]}_{features_type}.pkl'
    if (not cached or not os.path.exists(cache_file)):
        print('loading uncached data...')
        test_data = pd.read_csv(data_file,
                                sep='\t',
                                names=['inchikey', 'smiles', 'rt'])
        x, y = get_data(test_data, features_type)
        # split data into training testing + validation
        (train_x,
         train_y), (val_x, val_y), (test_x,
                                    test_y) = split_arrays(x, y, (0.2, 0.05))
    else:
        with open(cache_file, 'rb') as f:
            (train_x, train_y), (val_x, val_y), (test_x,
                                                 test_y) = pickle.load(f)
    if (write_cache):
        with open(cache_file, 'wb') as f:
            pickle.dump(((train_x, train_y), (val_x, val_y), (test_x, test_y)),
                        f)
    if (reduce_f):
        # remove unnecessary features
        train_x_new, removed = reduce_features(train_x)
        train_x = np.delete(train_x, removed, axis=1)
        val_x = np.delete(val_x, removed, axis=1)
        test_x = np.delete(test_x, removed, axis=1)
    if (standardize):
        # standardize data
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        val_x = scaler.transform(val_x)
        test_x = scaler.transform(test_x)
    if (n_features is not None):
        train_x = train_x[:, :n_features]
        val_x = val_x[:, :n_features]
        test_x = test_x[:, :n_features]
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


# def lgbmranker(train_x, train_y, val_x, val_y, test_x, test_y, ):
#     train_q, val_q, test_q = [train_x.shape[0]], [val_x.shape[0]], [test_x.shape[0]]
#     max_ = max(*train_y, *val_y, *test_y)
#     gbm = lgb.LGBMRanker(label_gain=[i ** 2 for i in range(max_ + 1)], n_estimators=100)
#     gbm.fit(train_x, train_y, group=train_q, eval_set=[(val_x, val_y)],
#             eval_group=[val_q], verbose=True)
#     preds = gbm.predict(test_x)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Retention Order index prediction')
    parser.add_argument('-i', '--input', default=['example.csv'], help='Either CSV or dataset IDs',
                        nargs='+')
    parser.add_argument('-t', '--type', help='type of features',
                        default='rdk',
                        choices=['rdk', 'mordred', 'morgan3', '3d', 'all'])
    parser.add_argument('--test', help='dataset to test on',
                        default=[], nargs='+')
    parser.add_argument('-f', '--features', default=[], help='custom features',
                        nargs='+')
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--reduce_features', action='store_true')
    parser.add_argument('-b', '--batch_size', default=1024, type=int,
                        help=' ')
    parser.add_argument('--cclasses', action='store_true', help='use classyfire '
                        'compound classes as additional features if available')
    parser.add_argument('--sysinfo', action='store_true', help='use column information '
                        'as additional features if available')
    parser.add_argument('--usp_codes', action='store_true', help='use column usp codes '
                        'as onehot system features (only if `--sysinfo` is set)')
    parser.add_argument('--isomeric', action='store_true', help=' ')
    parser.add_argument('--repo_root_folder', default='/home/fleming/Documents/Projects/RtPredTrainingData/',
                        help='location of the dataset github repository')
    parser.add_argument('-e', '--epochs', default=10, type=int, help=' ')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--no_bar', action='store_true', help='no progress-bar')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--sizes',
                        type=int,
                        nargs='+',
                        help='hidden layer sizes',
                        default=[10, 10])
    parser.add_argument('--pair_step', default=1, type=int, help=' ')
    parser.add_argument('--pair_stop', default=None, type=int, help=' ')
    parser.add_argument('--num_features', default=None, type=int, help=' ')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help=' ')
    parser.add_argument('--classes_l_thr', default=0.005, type=float, help=' ')
    parser.add_argument('--classes_u_thr', default=0.25, type=float, help=' ')
    parser.add_argument('--void_rt', default=0.0, type=float, help=' ')
    parser.add_argument('--cache_file', default='cached_descs.pkl', help=' ')
    parser.add_argument('--run_name', default=None, help=' ')
    parser.add_argument('--export_rois', action='store_true', help=' ')
    parser.add_argument('--use_weights', action='store_true', help=' ')
    parser.add_argument('--debug_onehot_sys', action='store_true', help=' ')
    parser.add_argument('--device', default=None,
                        help='can be `mirrored`, a specific device name like `gpu:1` '
                        'or `None` which automatically selects an option')
    parser.add_argument('--epsilon', type=float, default=1.,
                        help='difference in evaluation measure below which to ignore falsely predicted pairs')
    parser.add_argument('--columns_use_hsm', action='store_true', help=' ')
    parser.add_argument('--columns_hsm_data', default=
                        '/home/fleming/Documents/Projects/RtPredTrainingData/hsm.tsv',
                        help=' ')
    parser.add_argument('--custom_column_fields', default=None,
                        nargs='*', help=' ')
    parser.add_argument('--remove_train_compounds', action='store_true', help=' ')
    parser.add_argument('--remove_train_compounds_mode', default='all',
                        choices=['all', 'column', 'threshold'], help=' ')
    return parser.parse_args() if args is None else parser.parse_args(args)


def multi_eval_(bgs, its=3, ranker_args={
        'hidden_layer_sizes': (3, 3), 'activation': ('relu', 'relu'),
        'solver': 'adam'}, batch_size=2048, epochs=2):
    for bg in bgs:
        print(f'running with {bg.name}')
        bg_accs = []
        for it in range(its):
            print(f'{it}')
            ranker = RankNetNN(input_size=bg.train_x.shape[1],
                               **ranker_args)
            ranker.model.fit(bg,
                         epochs=epochs,
                         verbose=1,
                         validation_data=BatchGenerator(bg.val_x, bg.val_y,
                                                        batch_size))
            accs = [eval_(bg.train_y, ranker.predict(bg.train_x)),
                    eval_(bg.test_y, ranker.predict(bg.test_x)),
                    eval_(bg.val_y, ranker.predict(bg.val_x))]
            bg_accs.append(accs)
        print(f'data {bg.name}:')
        pprint(bg_accs)
        stats = ''
        for i, n in enumerate(['train', 'test', 'val']):
            stats += f'\t{n}: \t\t'
            for fn in (np.mean, np.median, min, max):
                stats += f'{fn.__name__} \t{fn([a[i] for a in bg_accs]):.3f} \t'
            stats += '\n'
        print(stats.rstrip())


def run_multi(args):
    bgs = []
    for kind in ['rdk', 'mordred']:
        print(f'generating {kind} data ...')
        data = Data.from_raw_file(args.input[0])
        data.cache_file = args.cache_file
        # cache_file = f'{splitext(basename(args.input[0]))[0]}_{kind}.pkl'
        # data.features_from_cache(cache_file)
        data.split_data()
        data.standardize()
        data.reduce_f()
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = data.get_split_data()
        bg = BatchGenerator(train_x, train_y, args.batch_size, use_weights=args.use_weights)
        bg.train_x = train_x
        bg.train_y = train_y
        bg.val_x = val_x
        bg.val_y = val_y
        bg.test_x = test_x
        bg.test_y = test_y
        bg.name =  'kind'
        bgs.append(bg)
    multi_eval_(bgs)


def predict(X, ranker, batch_size):
    preds = []
    ranker_output = K.function([ranker.model.layers[0].input], [ranker.model.layers[-3].get_output_at(0)])
    for x in np.array_split(X, np.ceil(X.shape[0] / batch_size * 10)):
        preds.append(ranker_output([x])[0].ravel())
    return np.concatenate(preds)


# def directr():
#      r = directRanker(max_steps=2)
#      d = Data.from_raw_file('test.csv')
#      d.features_from_cache('test_rdk.pkl')
#      d.split_data()
#      d.standardize()
#      (train_x, train_y), (val_x, val_y), (test_x, test_y) = d.get_split_data()
#      _x = val_x[:10]
#      _y = val_y[:10]
#      r.fit([_x], [_y], ranking=True)
#      r.predict_proba(_x)
#      eval_(_y, np.concatenate(r.predict_proba(_x)))

def export_predictions(data, preds, out, mode='all'):
    if (mode == 'all'):
        df = pd.DataFrame(data.df.iloc[np.concatenate((data.train_indices, data.test_indices, data.val_indices))])
    elif (mode == 'test'):
        df = pd.DataFrame(data.df.iloc[data.test_indices])
    else:
        raise NotImplementedError(mode)
    df['roi'] = preds
    df[['smiles', 'rt', 'roi']].to_csv(out, sep='\t', index=False, header=False)


if __name__ == '__main__':
    if '__file__' in globals():
        args = parse_arguments()
    else:
        # args = parse_arguments('-i 0045 0019 0063 0047 0017 0062 0024 0064 0048 0068 0086 0091 0096 0097 0080 0085 0087 0088 0098 0095 0100 0099 0077 0138 0179 0181 0182 0076 0084 0089 0090 -t rdk -e 100 -b 65536 --sizes 64 64 --standardize --sysinfo --balance --cclasses --classes_u_thr 0.8 --classes_l_thr 0.0005'.split())
        # args = parse_arguments('-i 0033 -t rdk -e 50 -b 131072 --standardize --sizes 256 256 --use_weights --sysinfo'.split())
        # args = parse_arguments('-i 0006 0037 0068 0117 -t rdk -e 50 -b 131072 --standardize --sizes 256 256 --use_weights --sysinfo -f MolLogP -t custom'.split())
        args = parse_arguments('-i 0024 0089 -t rdk -e 50 -b 131072 --standardize --sizes 256 256 --use_weights --sysinfo'.split())
    if (args.type == 'all'):
        args.type = None        # type ~= filter
    if (args.verbose):
        print(args)
    else:
        rdBase.DisableLog('rdApp.warning')
    if (args.cache_file is not None):
        features.write_cache = False # flag for reporting changes to cache
        if (args.verbose):
            print('reading in cache...')
        if (os.path.exists(args.cache_file)):
            features.cached = pickle.load(open(args.cache_file, 'rb'))
        else:
            features.cached = {}
            if (args.verbose):
                print('cache file does not exist yet')
    if (args.verbose):
        print('reading in data and computing features...')
    if (len(args.input) == 1 and os.path.exists(args.input[0])):
        data = Data.from_raw_file(args.input[0], void_rt=args.void_rt)
    elif (all(re.match(r'\d{4}', i) for i in args.input)):
        # dataset IDs
        data = Data(use_compound_classes=args.cclasses, use_system_information=args.sysinfo,
                    classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr,
                    use_usp_codes=args.usp_codes, custom_features=args.features,
                    use_hsm=args.columns_use_hsm, hsm_data=args.columns_hsm_data,
                    custom_column_fields=args.custom_column_fields)
        for did in args.input:
            data.add_dataset_id(did,
                                repo_root_folder=args.repo_root_folder,
                                void_rt=args.void_rt,
                                isomeric=args.isomeric)
        if (args.balance and len(args.input) > 1):
            data.balance()
        if (args.verbose):
            print('added data for datasets:')
            print('\n'.join([f'  - {did} ({name})' for did, name in
                             set(data.df[['dataset_id', 'column.name']].itertuples(index=False))]))
    else:
        raise Exception(f'input {args.input} not supported')
    data.compute_features(filter_features=args.type, n_thr=args.num_features, verbose=args.verbose)
    if args.debug_onehot_sys:
        sorted_dataset_ids = sorted(set(args.input) | set(args.test))
        data.compute_system_information(True, sorted_dataset_ids)
    if (args.verbose):
        print('done. preprocessing...')
    data.split_data()
    if (args.standardize):
        data.standardize()
    if (args.reduce_features):
        data.reduce_f()
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = data.get_split_data()
    if (args.verbose):
        print('done. Initializing BatchGenerator...')
    bg = BatchGenerator(train_x, train_y, args.batch_size, pair_step=args.pair_step,
                        pair_stop=args.pair_stop, use_weights=args.use_weights)
    if (args.device is not None and re.match(r'[cg]pu:\d', args.device.lower())):
        print(f'attempting to use device {args.device}')
        strategy = tf.distribute.OneDeviceStrategy(f'/{args.device.lower()}')
        context = strategy.scope()
    elif (len([dev for dev in tf.config.get_visible_devices() if dev.device_type == 'GPU']) > 1
        or args.device == 'mirrored'):
        # more than one gpu -> MirroredStrategy
        print('Using MirroredStrategy')
        strategy = tf.distribute.MirroredStrategy()
        context = strategy.scope()
    else:
        context = contextlib.nullcontext()
    if (args.verbose):
        print('done. Creating model...')
    with context:
        v = tf.Variable(1.0)
        if (args.verbose):
            print(f'using {v.device}')
        ranker = RankNetNN(input_size=train_x.shape[1],
                           hidden_layer_sizes=args.sizes,
                           activation=(['relu'] * len(args.sizes)),
                           solver='adam',
                           dropout_rate=args.dropout_rate)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                          restore_best_weights=True)
    if (args.verbose):
        ranker.model.summary()
        print('done. Training...')
    try:
        ranker.model.fit(bg,
                         callbacks=[es,
                                    # tf.keras.callbacks.TensorBoard(update_freq='epoch', histogram_freq=1,)
                                ],
                         epochs=args.epochs,
                         verbose=1 if not args.no_bar else 2,
                         validation_data=BatchGenerator(
                             val_x, val_y, args.batch_size, pair_step=args.pair_step,
                             pair_stop=args.pair_stop, use_weights=args.use_weights))
    except KeyboardInterrupt:
        print('interrupted training, evaluating...')
    print(f'train: {eval_(train_y, predict(train_x, ranker, args.batch_size), args.epsilon):.3f}')
    test_preds = predict(test_x, ranker, args.batch_size)
    print(f'test: {eval_(test_y, test_preds, args.epsilon):.3f}')
    print(f'val: {eval_(val_y, predict(val_x, ranker, args.batch_size), args.epsilon):.3f}')
    if (args.export_rois):
        if (args.run_name is None):
            from datetime import datetime
            time_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
            run_name = f'ranknet_{time_str}'
        else:
            run_name = args.run_name
        if not os.path.isdir('runs'):
            os.mkdir('runs')
        export_predictions(data, test_preds, f'runs/{run_name}_test.tsv', 'test')
    if (args.balance and len(args.input) > 1):  # ===LEFT-OUT EVAL===
        print('evaluating on data left-out when balancing')
        for ds in args.input:
            d = Data(use_compound_classes=args.cclasses, use_system_information=args.sysinfo,
                     classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr,
                     use_usp_codes=args.usp_codes, custom_features=args.features,
                     use_hsm=args.columns_use_hsm, hsm_data=args.columns_hsm_data,
                     custom_column_fields=data.custom_column_fields, columns_remove_na=False)
            d.add_dataset_id(ds,
                             repo_root_folder=args.repo_root_folder,
                             void_rt=args.void_rt,
                             isomeric=args.isomeric)
            perc = len(d.df.loc[d.df.id.isin(data.heldout.id)]) / len(d.df)
            d.df.drop(d.df.loc[~d.df.id.isin(data.heldout.id)].index, inplace=True)
            if (len(d.df) == 0):
                print(f'no data left for {ds}')
                continue
            d.compute_features(filter_features=args.type, n_thr=args.num_features, verbose=args.verbose)
            if args.debug_onehot_sys:
                d.compute_system_information(True, sorted_dataset_ids, use_usp_codes=args.usp_codes)
            d.split_data()
            if (args.standardize):
                d.standardize(data.scaler)
            if (args.reduce_features):
                d.reduce_f()
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = d.get_split_data()
            X = np.concatenate((train_x, test_x, val_x))
            Y = np.concatenate((train_y, test_y, val_y))
            preds = predict(X, ranker, args.batch_size)
            print(f'{ds}: {eval_(Y, preds, args.epsilon):.3f} \t (#data: {len(Y)}, held-out percentage: {perc:.2f})')
            if (args.export_rois):
                export_predictions(d, preds, f'runs/{run_name}_heldout_{ds}.tsv')
    if (len(args.test) > 0):    # ===TESTING===
        print(f'evaluating on different dataset(s) ({args.test})')
        for ds in args.test:
            d = Data(use_compound_classes=args.cclasses, use_system_information=args.sysinfo,
                     classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr,
                     use_usp_codes=args.usp_codes, custom_features=args.features,
                     use_hsm=args.columns_use_hsm, hsm_data=args.columns_hsm_data,
                     custom_column_fields=data.custom_column_fields, columns_remove_na=False)
            d.add_dataset_id(ds,
                             repo_root_folder=args.repo_root_folder,
                             void_rt=args.void_rt,
                             isomeric=args.isomeric)
            if (args.remove_train_compounds):
                if (args.remove_train_compounds_mode == 'all'):
                    train_compounds = set(data.df['inchi.std'])
                elif (args.remove_train_compounds_mode == 'column'):
                    this_column = d.df['column.name'].values[0]
                    train_compounds = set(data.df.loc[data.df['column.name'] == this_column, 'inchi.std'])
                else:
                    raise NotImplementedError(args.remove_train_compounds_mode)
                prev_len = len(d.df)
                d.df = d.df.loc[~d.df['inchi.std'].isin(train_compounds)]
                if args.verbose:
                    print(f'{ds} evaluation: removed {prev_len - len(d.df)} compounds also appearing '
                          f'in the training data (now {len(d.df)} compounds)')
                if (len(d.df) < 2):
                    print(f'too few compounds ({len(d.df)}), skipping ...')
                    continue
            d.compute_features(filter_features=args.type, n_thr=args.num_features, verbose=args.verbose)
            if args.debug_onehot_sys:
                d.compute_system_information(True, sorted_dataset_ids, use_usp_codes=args.usp_codes)
            d.split_data()
            if (args.standardize):
                d.standardize(data.scaler)
            if (args.reduce_features):
                d.reduce_f()
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = d.get_split_data()
            X = np.concatenate((train_x, test_x, val_x))
            Y = np.concatenate((train_y, test_y, val_y))
            preds = predict(X, ranker, args.batch_size)
            print(f'{ds}: {eval_(Y, preds, args.epsilon):.3f} \t (#data: {len(Y)})')
            if (args.export_rois):
                export_predictions(d, preds, f'runs/{run_name}_{ds}.tsv')
    if (args.cache_file is not None and features.write_cache):
        if (args.verbose):
            print('writing cache, don\'t interrupt!!')
        pickle.dump(features.cached, open(args.cache_file, 'wb'))
