from argparse import Namespace
from itertools import combinations, product
from random import sample, shuffle
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
from typing import Optional, List, Tuple, Union, Iterable, Callable, Literal
import logging
import sys
from pprint import pformat
from time import time
from datetime import timedelta
import json

from features import features

logger = logging.getLogger('rtranknet.utils')
info = logger.info
warning = logger.warning

REL_COLUMNS = ['column.length', 'column.id', 'column.particle.size', 'column.temperature',
               'column.flowrate', 'eluent.A.h2o', 'eluent.A.meoh', 'eluent.A.acn',
               'eluent.A.formic', 'eluent.A.nh4ac', 'eluent.A.nh4form',
               'eluent.B.h2o', 'eluent.B.meoh', 'eluent.B.acn', 'eluent.B.formic',
               'eluent.B.nh4ac', 'eluent.B.nh4form', 'gradient.start.A',
               'gradient.end.A']
REL_ONEHOT_COLUMNS = ['class.pH.A', 'class.pH.B', 'class.solvent']


def csr2tf(csr):
    indices = []
    values = []
    for (i, j), v in csr.todok().items():
        indices.append([i, j])
        values.append(v)
    return tf.sparse.SparseTensor(indices, values, csr.shape)

def weight_stats(pkl, confl=[]):
    x = pickle.load(open(pkl, 'rb'))
    confl_weights = [_[2] for _ in x if _[0] in confl]
    nonconfl_weights = [_[2] for _ in x if _[0] not in confl]
    print(pd.DataFrame({'nonconfl': nonconfl_weights}).describe())
    print(pd.DataFrame({'confl': confl_weights}).describe())

def rt_diff_weight_fun(rt_diff, weight=1, a=20, b=0.75):
    return (weight              # upper asymptote
            / (1 +
               np.exp(-(
                   a)          # slope
                      * (rt_diff - b))) ** (1))

def pair_weights(smiles1: str, smiles2: str, rt_diff: float,
                 nr_group_pairs: int, nr_group_pairs_max: int,
                 confl_weights_modifier: float, confl_pair_list: Iterable[frozenset]=[],
                 cutoff:float=1e-4, only_confl=False, weight_steepness=20,
                 weight_mid=0.75) -> Optional[float]:
    # group (~dataset) size balancing modifier
    base_weight = nr_group_pairs_max / nr_group_pairs # roughly between 1 and 500
    # conflicting (-> important) pair modifier
    if (frozenset([smiles1, smiles2]) in confl_pair_list):
        base_weight *= confl_weights_modifier
    elif only_confl:
        base_weight = 0
    # rt diff weight modifier
    base_weight = rt_diff_weight_fun(rt_diff, base_weight, a=weight_steepness, b=weight_mid)
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


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, ids=None, batch_size=32, shuffle=True,
                 pair_step=1, pair_stop=None, dataset_info=None,
                 void_info=None, no_inter_pairs=False, no_intra_pairs=False,
                 max_indices_size=None,
                 use_weights=True, use_group_weights=True,
                 weight_steep=4, weight_mid=0.75,
                 void=None, y_neg=False, multix=False,
                 conflicting_smiles_pairs=[]):
        self.x = x              # (a) descriptors (b) graphs (c) descriptors + graphs
        self.y = y              # retention times
        self.ids = ids          # IDs (e.g., smiles) for every x/y pair
        self.multix = multix    # x: descriptors + graphs
        self.use_weights = use_weights # pair weights
        self.use_group_weights = use_group_weights # dataset weights
        self.weight_steep = weight_steep
        self.weight_mid = weight_mid
        self.pair_step = pair_step
        self.pair_stop = pair_stop
        self.dataset_info = dataset_info # list of dataset IDs for each x/y pair
        self.void_info = void_info       # void time for each dataset ID
        self.no_inter_pairs = no_inter_pairs # don't generate inter-dataset pairs
        self.no_intra_pairs = no_intra_pairs # don't generate within-dataset pairs
        self.max_indices_size = max_indices_size
        self.void = void        # global void time (for every dataset)
        self.y_neg = y_neg      # -1 for negative pairs (instead of 0)
        self.x1_indices, self.x2_indices, self.y_trans, self.weights = self._transform_pairwise(
            y, ids, dataset_info=dataset_info, void_info=void_info, no_inter_pairs=no_inter_pairs,
            no_intra_pairs=no_intra_pairs, max_indices_size=max_indices_size, use_group_weights=use_group_weights,
            conflicting_smiles_pairs=conflicting_smiles_pairs)
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

    @staticmethod
    def dataset_pair_it(indices, pair_step=1, pair_stop=None,
                        max_indices_size=None):
        n = len(indices)
        if (max_indices_size is None):
            it = range(n)
        else:
            it = sorted(sample(list(range(n)), min(max_indices_size, n)))
        for i in it:
            for j in range(i + 1,
                           (n if pair_stop is None else min(i + pair_stop, n)),
                           pair_step):
                yield indices[i], indices[j]

    @staticmethod
    def inter_dataset_pair_it(indices1, indices2, pair_step=1, pair_stop=None,
                              nr_groups_norm=1, max_indices_size=None):
        max_ = max(len(indices1), len(indices2))
        if (max_indices_size is not None):
            max_ = min(max_, max_indices_size)
        all_combs = list(product(indices1, indices2))
        k = (max_ * np.ceil((pair_stop if pair_stop is not None else max_) / pair_step)
             * nr_groups_norm).astype(int)
        return iter(sample(all_combs, min(k, len(all_combs))))

    @staticmethod
    def get_pair(y, i, j, void_i=0, void_j=0, y_neg=False):
        pos_idx, neg_idx = (i, j) if y[i] > y[j] else (j, i)
        # void
        if (y[i] < void_i and y[j] < void_j):
            # don't take pairs where both compounds are in void volume
            return None
        # balanced class
        if 1 != (-1)**(pos_idx + neg_idx):
            return pos_idx, neg_idx, 1
        else:
            return neg_idx, pos_idx, (-1 if y_neg else 0)

    def get_comparable_pairs(self, indices_i, indices_j, rts, ids,
                             void_i=0, void_j=0, y_neg=False, epsilon=0.5,
                             pairs_compute_threshold=None):
        pairs = set()
        def make_pairs(indices_pre, indices_post):
            for i, (i_pre, i_post) in enumerate(product(indices_pre, indices_post)):
                yield (i_post, i_pre, 1) if 1 == (-1)**i else (i_pre, i_post, -1 if y_neg else 0)
        inters = list(set([ids[i] for i in indices_i]) & set([ids[j] for j in indices_j]))
        shuffle(inters)
        # TODO: problem if IDs not unique, assert this somewhere!
        for id_k in inters:
            if (pairs_compute_threshold is not None and len(pairs) > pairs_compute_threshold):
                info('too many inter-pairs to consider; aborting with compute threshold')
                warning('inter-pairs might be unbalanced due to their potentially large number!')
                break
            k_i = [i for i in indices_i if ids[i] == id_k][0]
            k_j = [j for j in indices_j if ids[j] == id_k][0]
            if (rts[k_i] < void_i or rts[k_j] < void_j):
                continue
            pre_is = [i for i in indices_i if rts[i] + epsilon < rts[k_i] and rts[i] >= void_i]
            post_is = [i for i in indices_i if rts[i] > rts[k_i] + epsilon and rts[i] >= void_i]
            pre_js = [j for j in indices_j if rts[j] + epsilon < rts[k_j] and rts[j] >= void_j]
            post_js = [j for j in indices_j if rts[j] > rts[k_j] + epsilon and rts[j] >= void_j]
            pairs |= set(make_pairs(pre_is, post_js))
            pairs |= set(make_pairs(pre_js, post_is))
        return list(pairs)

    def _transform_pairwise(self, y, ids=None, dataset_info=None,
                            void_info=None, no_inter_pairs=False,
                            no_intra_pairs=False,
                            max_indices_size=None,
                            use_group_weights=True,
                            conflicting_smiles_pairs=[]):
        assert not (no_inter_pairs and no_intra_pairs), 'no_inter_pairs and no_intra_pairs can\'t be both active'
        if (ids is not None):
            assert len(y) == len(ids), 'list of IDs (e.g., smiles) must have same length as list of RTs'
        x1_indices = []
        x2_indices = []
        y_trans = []
        weights = []

        ###############
        # DEBUG START #
        ###############

        """
        pairs = pickle.load(open('/home/fleming/Documents/Uni/RTpred/pairs_0068_0138.pkl', 'rb'))
        from itertools import permutations, combinations
        for n, (i, j) in enumerate(combinations(range(len(y)), 2)):
            id1, id2 = ids[i], ids[j]
            if (dataset_info[i] != dataset_info[j]):
                continue
            if (frozenset([id1, id2]) not in pairs):
                continue
            if (np.abs(y[i] - y[j]) < 0.5):
                print('void continue', dataset_info[i], y[i], y[j])
                continue
            y_res = 1 if y[i] > y[j] else (-1 if self.y_neg else 0)
            if (-1**n == 1):
                x1_indices.append(i)
                x2_indices.append(j)
                y_trans.append(y_res)
            else:
                x1_indices.append(j)
                x2_indices.append(i)
                y_trans.append(1 if y_res != 1 else (-1 if self.y_neg else 0))
            weights.append(1.0)
        from time import time
        pickle.dump([(tuple([ids[x1_indices[i]], ids[x2_indices[i]]]), y_trans[i], dataset_info[x1_indices[i]])
                      for i in range(len(y_trans))],
                    open(f'/tmp/rtranknet_weights_dump_{int(time() * 1000)}.pkl', 'wb'))
        return np.asarray(x1_indices), np.asarray(x2_indices), np.asarray(
            y_trans), np.asarray(weights)

        """

        #############
        # DEBUG END #
        #############


        # group by dataset
        groups = {}
        pair_nrs = {}
        group_index_start = {}
        group_index_end = {}
        # confl_pair_report = {}
        if (dataset_info is None):
            groups['unk'] = list(range(len(y)))
        else:
            assert len(dataset_info) == len(y), f'{len(dataset_info)=} != {len(y)=}'
            for i in range(len(y)):
                groups.setdefault(dataset_info[i], []).append(i)
        # same-dataset pairs
        inter_pair_nr = intra_pair_nr = 0
        if (not no_intra_pairs):
            info('computing intra-dataset pairs...')
            t0 = time()
            for group in groups:
                group_index_start[group] = len(weights)
                group_void_rt = void_info[group] if void_info is not None and group in void_info else self.void
                pair_nr = 0
                it = BatchGenerator.dataset_pair_it(groups[group], self.pair_step, self.pair_stop,
                                                           max_indices_size=max_indices_size)
                # if (logger.level <= logging.INFO):
                #     from tqdm import tqdm
                #     it = tqdm(it)
                for i, j in it:
                    res = BatchGenerator.get_pair(y, i, j, group_void_rt or 0, group_void_rt or 0, self.y_neg)
                    if (res is None):
                        continue
                    pos_idx, neg_idx, yi = res
                    # debug
                    # if yi > 0:
                    #     print(f'intra {ids[pos_idx]} < {ids[neg_idx]}')
                    # else:
                    #     print(f'intra {ids[neg_idx]} < {ids[pos_idx]}')
                    x1_indices.append(pos_idx)
                    x2_indices.append(neg_idx)
                    y_trans.append(yi)
                    # weights
                    weights.append(1.0) # will be computed later when group weights are known
                    pair_nr += 1
                pair_nrs[group] = pair_nr
                intra_pair_nr += pair_nr
                group_index_end[group] = len(weights)
            info(f'done ({str(timedelta(seconds=time() - t0))} elapsed)')
        # between groups
        if (not no_inter_pairs):
            info('compute inter dataset pairs...')
            t0 = time()
            inter_group_nr = len(list(combinations(groups, 2)))
            it = combinations(groups, 2)
            if (logger.level <= logging.INFO):
                    from tqdm import tqdm
                    it = tqdm(list(it))
            for group1, group2 in it:
                group_index_start[(group1, group2)] = len(weights)
                void_i = void_info[group1] if void_info is not None and group1 in void_info else self.void
                void_j = void_info[group2] if void_info is not None and group2 in void_info else self.void
                pair_nr = 0
                n = min(max(len(groups[group1]), len(groups[group2])), max_indices_size or 1e9)
                max_pair_nr = (n * np.ceil((self.pair_stop if self.pair_stop is not None else n) / self.pair_step)
                               * (1/(inter_group_nr / len(groups)))).astype(int)
                potential_pairs = self.get_comparable_pairs(groups[group1], groups[group2], y, ids,
                                                            void_i=void_i or 0, void_j=void_j or 0,
                                                            y_neg=self.y_neg, epsilon=0.5,
                                                            pairs_compute_threshold=10 * max_pair_nr)
                # debug
                # for i, j, yi in potential_pairs:
                #     if yi > 0:
                #         print(f'{ids[i]} < {ids[j]}')
                #     else:
                #         print(f'{ids[j]} < {ids[i]}')
                print(f'{group1}, {group2} {max_pair_nr=}, {(len(potential_pairs))=}')
                for pos_idx, neg_idx, yi in iter(sample(potential_pairs, min(max_pair_nr, len(potential_pairs)))):
                    x1_indices.append(pos_idx)
                    x2_indices.append(neg_idx)
                    y_trans.append(yi)
                    weights.append(1.0) # absolute rt difference of pairs of two different datasets can't be compared
                    pair_nr += 1
                pair_nrs[(group1, group2)] = pair_nr
                inter_pair_nr += pair_nr
                group_index_end[(group1, group2)] = len(weights)
            info(f'done ({str(timedelta(seconds=time() - t0))} elapsed)')
        print(f'{inter_pair_nr=}, {intra_pair_nr=}')
        all_groups_list = list(pair_nrs)
        print(pd.DataFrame({'group': all_groups_list, 'pair numbers':
                            [pair_nrs[g] for g in all_groups_list]}).describe())
        nr_group_pairs_max = max(list(pair_nrs.values()) + [0])
        info('computing pair weights')
        for g in pair_nrs:
            # weight_modifier = 100 TODO:
            weight_modifier = 10
            for i in range(group_index_start[g], group_index_end[g]):
                rt_diff = (1e8 if isinstance(g, tuple) # no statement can be made on rt diff for inter-group pairs
                           else np.abs(y[x1_indices[i]] - y[x2_indices[i]]))
                weights[i] = pair_weights(ids[x1_indices[i]], ids[x2_indices[i]], rt_diff,
                                          pair_nrs[g] if use_group_weights else nr_group_pairs_max,
                                          nr_group_pairs_max, weight_modifier, conflicting_smiles_pairs,
                                          only_confl=False)
                # NOTE: pair weights can be "None"
        info('done. removing None weights')
        # remove Nones
        x1_indices_new = []
        x2_indices_new = []
        y_trans_new = []
        weights_new = []
        removed_counter = 0
        for i in range (len(y_trans)):
            if (weights[i] is not None):
                x1_indices_new.append(x1_indices[i])
                x2_indices_new.append(x2_indices[i])
                y_trans_new.append(y_trans[i])
                weights_new.append(weights[i])
            else:
                removed_counter += 1
        print(f'removed {removed_counter} (of {len(y_trans)}) pairs for having "None" weights')
        # if (use_group_weights):
        #     # group weights: intra_pair balanced and inter_pair balanced individually
        #     group_weights = {}
        #     info(f'{inter_pair_nr=}, {intra_pair_nr=}')
        #     first_inter = first_intra = True
        #     for group in pair_nrs:
        #         if pair_nrs[group] == 0:
        #             group_weights[group] = 1.0
        #             continue
        #         if isinstance(group, tuple): # same overall weight on inter vs intra weights
        #             group_weights[group] = inter_pair_nr / pair_nrs[group] / len(groups)
        #             if (first_inter):
        #                 info(f'inter group weights * nr_pairs = {group_weights[group] * pair_nrs[group]}')
        #                 first_inter = False
        #         else:
        #             group_weights[group] = intra_pair_nr / pair_nrs[group]
        #             if (first_intra):
        #                 info(f'intra group weights * nr_pairs = {group_weights[group] * pair_nrs[group]}')
        #                 first_intra = False
        #     weights = np.asarray(weights)
        #     # multiply rt_diff weights with group weights
        #     for groups in group_index_start:
        #         start, end = group_index_start[groups], group_index_end[groups]
        #         g_weight = group_weights[groups]
        #         weights[start:end] *= g_weight
        #     if (use_conflict_weights):
        #         # drastically up-weigh pairs which have different orders depending on the column
        #         for groups in group_index_start: # inter and intra?
        #             start, end = group_index_start[groups], group_index_end[groups]
        #             all_pair_weights = np.sum(weights[start:end])
        #             # how many conflicting pairs are there in this group?
        #             confl_pairs = [i for i in range(start, end)
        #                            if frozenset([ids[x1_indices[i]], ids[x2_indices[i]]]) in conflicting_smiles_pairs]
        #             if (len(confl_pairs) > 0):
        #                 # w(non_confl_pairs) == w(confl_pairs)
        #                 # confl_weights = all_pair_weights / len(confl_pairs)
        #                 confl_weights = 100
        #                 for i in confl_pairs:
        #                     weights[i] *= confl_weights
        #                 confl_pair_report[groups] = len(confl_pairs)
        #         print('# conflicting pairs detected per dataset(-pair):', pformat(confl_pair_report))
        # debug dump
        # from time import time
        # pickle.dump([[(frozenset([ids[x1_indices[i]], ids[x2_indices[i]]]), y_trans[i], weights[i])
        #               for i in range(len(y_trans))],
        #              group_index_start, group_index_end],
        #             open(f'/tmp/rtranknet_weights_dump_{int(time() * 1000)}.pkl', 'wb'))
        info('done generating pairs')
        return np.asarray(x1_indices_new), np.asarray(x2_indices_new), np.asarray(
            y_trans_new), np.asarray(weights_new)

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
                        hsm = hsm.append(fallback)
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
                        tanaka = tanaka.append(fallback)
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
            del column_information['id']
            # column_information.set_index('dataset_id', inplace=True, drop=False)
            if (self.use_newonehot):
                # include pH info (TODO: should be included in metadata)
                if (self.ph is None):
                    # only read in once
                    self.ph = pd.read_csv(os.path.join(repo_root_folder, 'ph_info.csv'), sep='\t', index_col=0)[REL_ONEHOT_COLUMNS]
                    # one-hot columns:
                    for c in REL_ONEHOT_COLUMNS:
                        self.ph = pd.merge(left=self.ph, right=pd.get_dummies(self.ph[c], prefix=c),
                                           left_index=True, right_index=True).drop(columns=c)
                    self.ph.columns = self.ph.columns.str.replace(' ', '') # remove space from col names
                column_information = column_information.set_index('dataset_id').join(
                    self.ph, rsuffix='excel')
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
