from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union, Any
from torch.utils.data import Dataset
import numpy as np
import logging
from logging import info, warning
from time import time
from datetime import timedelta
from itertools import combinations, product
from random import sample, shuffle
from utils import pair_weights
import pandas as pd
from collections import Counter, defaultdict
from pprint import pprint

logger = logging.getLogger('rtranknet.utils')
info = logger.info
warning = logger.warning


@dataclass
class RankDataset(Dataset):
    x_mols: List[Union[Any, str]]       # mol graphs or SMILES
    x_extra: Union[np.ndarray, List[List[float]]] # extra compound features, e.g., logp
    x_sys: Union[np.ndarray, List[List[float]]]   # system features
    x_ids: List[str]                              # ID (e.g., smiles) for each sample
    y: Union[np.ndarray, List[float]]             # retention times
    x_sys_global_num: Optional[int] = None        # which (exclusive max slice index) of the x_sys features are global for the whole dataset
    use_pair_weights: bool=True                   # use pair weights (epsilon)
    epsilon: float=0.5                                  # soft threshold for retention time difference
    discard_smaller_than_epsilon: bool=False      # don't weigh by rt diff; simply discard any pairs with rt_diff<epsilon
    use_group_weights: bool=True                  # weigh number of samples per group
    cluster: bool=False                           # cluster datasets with same column params for calculating
                                                  # group weights
    downsample_groups: bool=False                 # min number of pairs will be used as the max pair nr for each group
    downsample_always_confl: bool=False           # include all conflicting pairs also when downsampling
    downsample_factor: float=1.0                  # if greater than 1, some clusters may have less pairs
    group_weights_only_intra_cluster: bool=False  # group-weights are used, but only for weighing within a cluster
    weight_steepness: float=20                    # steepness of the pair_weight_fn
    weight_mid: float=0.75                        # mid-factor of the weight_mid
    dynamic_weights: bool=True                    # adapt epsilon to gradient length
    pair_step: int=1                              # step size for generating pairs
    pair_stop: Optional[int]=None                 # stop number for generating pairs
    dataset_info: Optional[List[str]] = None      # Dataset ID for each datum
    void_info: Optional[Dict[str, float]] = None  # void time mapping for dataset IDs
    void: Optional[float]=None                    # global void time
    no_inter_pairs: bool=True                     # don't generate inter dataset pairs
    no_intra_pairs: bool=False                    # don't generate intra dataset pairs
    max_indices_size:Optional[int]=None           # limit for the size of indices
    max_num_pairs:Optional[int]=None              # limit for the number of pairs per dataset/group
    y_neg : bool=False                            # -1 instead of 0 for negative pair
    y_float : bool=False                          # yield target values as floats instead of as longs
    conflicting_smiles_pairs:dict = field(default_factory=dict) # conflicting pairs (smiles)
    only_confl: bool=False                                          # gather only conflicting pairs
    confl_weight: float=1.                                          # weight modifier for conflicting pairs

    def __post_init__(self):
        if (isinstance(self.x_extra, np.ndarray)):
            self.x_extra = self.x_extra.astype('float32')
        if (isinstance(self.x_sys, np.ndarray)):
            self.x_sys = self.x_sys.astype('float32')
        # assert dimensions etc.
        assert len(self.x_mols) == len(self.x_extra) == len(self.x_sys) == len(self.x_ids) == len(self.y)
        if (self.dataset_info is not None):
            assert len(self.y) == len(self.dataset_info)
        assert not (self.no_inter_pairs and self.no_intra_pairs), (
            'no_inter_pairs and no_intra_pairs can\'t be both set')
        # preprocess doublets
        self.preprocess_doublets()
        # transform single compounds(+info) into pairs for ranking
        transformed = self._transform_pairwise()
        self.x1_indices = transformed['x1_indices']
        self.x2_indices = transformed['x2_indices']
        self.y_trans = transformed['y_trans']
        if (self.y_float):
            self.y_trans = self.y_trans.astype('float32')
        self.weights = transformed['weights']
        self.is_confl = transformed['is_confl']

    def _transform_pairwise(self):
        x1_indices = []
        x2_indices = []
        y_trans = []
        weights = []
        is_confl = []
        # group by dataset
        groups = {}
        pair_nrs = {}
        group_index_start = {}
        group_index_end = {}
        groups_max_rts = defaultdict(float)
        # confl_pair_report = {}
        if (self.dataset_info is None):
            groups['unk'] = list(range(len(self.y)))
        else:
            for i in range(len(self.y)):
                groups.setdefault(self.dataset_info[i], []).append(i)
                groups_max_rts[self.dataset_info[i]] = max(groups_max_rts[self.dataset_info[i]],
                                                           self.y[i])
        print(f'{groups_max_rts=}')
        # preprocess confl pair list for O(1) lookup
        # and disregard confl pairs not conflicting for this training set
        confl_pairs_lookup = {k for k, v in self.conflicting_smiles_pairs.items()
                              if any(all(xi in groups for xi in x) for x in v)}
        print(f'using {len(confl_pairs_lookup)} out of the {len(self.conflicting_smiles_pairs)} '
              'conflicting pairs provided')
        # same-dataset pairs
        inter_pair_nr = intra_pair_nr = 0
        if (not self.no_intra_pairs):
            info('computing intra-dataset pairs...')
            t0 = time()
            for group in groups:
                group_index_start[group] = len(weights)
                group_void_rt = (self.void_info[group] if self.void_info is not None
                                 and group in self.void_info else self.void)
                pair_nr = 0
                # get conflicting smiles pairs indices
                confl_indices = set()
                if (len(confl_pairs_lookup) > 0):
                    for i, j in combinations(groups[group], 2):
                        if frozenset((self.x_ids[i], self.x_ids[j])) in confl_pairs_lookup:
                            confl_indices.add(frozenset((i, j)))
                it = self.dataset_pair_it(groups[group], self.pair_step, self.pair_stop,
                                          max_indices_size=self.max_indices_size,
                                          max_num_pairs=self.max_num_pairs,
                                          obl_indices=confl_indices)
                if (logger.level <= logging.INFO):
                    from tqdm import tqdm
                    it = tqdm(it)
                doublets_filtered = 0
                for i, j, w in it:
                    # filter out invalid pairs due to doublets
                    if (hasattr(self, 'doublet_rt_ranges') and
                        ((group, self.x_ids[i]) in self.doublet_rt_ranges or
                         (group, self.x_ids[j]) in self.doublet_rt_ranges)):
                        min_i = (self.doublet_rt_ranges[(group, self.x_ids[i])][0]
                                 if (group, self.x_ids[i]) in self.doublet_rt_ranges
                                 else self.y[i])
                        max_i = (self.doublet_rt_ranges[(group, self.x_ids[i])][1]
                                 if (group, self.x_ids[i]) in self.doublet_rt_ranges
                                 else self.y[i])
                        min_j = (self.doublet_rt_ranges[(group, self.x_ids[j])][0]
                                 if (group, self.x_ids[j]) in self.doublet_rt_ranges
                                 else self.y[j])
                        max_j = (self.doublet_rt_ranges[(group, self.x_ids[j])][1]
                                 if (group, self.x_ids[j]) in self.doublet_rt_ranges
                                 else self.y[j])
                        if (((max_i >= min_j) and (max_i <= max_j))
                            or ((min_i >= min_j) and (min_i <= max_j))
                            or ((min_i >= min_j) and (max_i <= max_j))
                            or ((max_i >= max_j) and (min_i <= min_j))):
                            # print(f'filtered doublet pair ({self.x_ids[i]}, {self.x_ids[j]}); ranges '
                            #       f'{(min_i, max_i)}, {(min_j, max_j)}')
                            doublets_filtered += 1
                            continue
                    res = self.get_pair(self.y, i, j, group_void_rt or 0, group_void_rt or 0, self.y_neg)
                    if (res is None):
                        continue
                    pos_idx, neg_idx, yi = res
                    x1_indices.append(pos_idx)
                    x2_indices.append(neg_idx)
                    y_trans.append(yi)
                    # weights
                    weights.append(w)
                    # is conflicting pair?
                    is_confl.append(frozenset((pos_idx, neg_idx)) in confl_indices)
                    pair_nr += 1
                pair_nrs[group] = pair_nr
                intra_pair_nr += pair_nr
                group_index_end[group] = len(weights)
                info(f'filtered out {doublets_filtered} invalid pairs due to doublets for group {group}')
            info(f'done ({str(timedelta(seconds=time() - t0))} elapsed)')
        # between groups
        if (not self.no_inter_pairs):
            info('compute inter dataset pairs...')
            t0 = time()
            inter_group_nr = len(list(combinations(groups, 2)))
            it = combinations(groups, 2)
            if (logger.level <= logging.INFO):
                    from tqdm import tqdm
                    it = tqdm(list(it))
            for group1, group2 in it:
                group_index_start[(group1, group2)] = len(weights)
                void_i = (self.void_info[group1] if self.void_info is not None
                          and group1 in self.void_info else self.void)
                void_j = (self.void_info[group2] if self.void_info is not None
                          and group2 in self.void_info else self.void)
                pair_nr = 0
                n = min(max(len(groups[group1]), len(groups[group2])), self.max_indices_size or 1e9)
                max_pair_nr = (n * np.ceil((self.pair_stop if self.pair_stop is not None else n) / self.pair_step)
                               * (1/(inter_group_nr / len(groups)))).astype(int)
                potential_pairs = self.get_comparable_pairs(groups[group1], groups[group2], self.y, self.x_ids,
                                                            void_i=void_i or 0, void_j=void_j or 0,
                                                            y_neg=self.y_neg, epsilon=self.epsilon,
                                                            pairs_compute_threshold=10 * max_pair_nr)
                info(f'{group1}, {group2} {max_pair_nr=}, {(len(potential_pairs))=}')
                for pos_idx, neg_idx, yi in iter(sample(potential_pairs, min(max_pair_nr, len(potential_pairs)))):
                    x1_indices.append(pos_idx)
                    x2_indices.append(neg_idx)
                    y_trans.append(yi)
                    weights.append(1.0) # absolute rt difference of pairs of two different datasets can't be compared
                    is_confl.append(None)
                    pair_nr += 1
                pair_nrs[(group1, group2)] = pair_nr
                inter_pair_nr += pair_nr
                group_index_end[(group1, group2)] = len(weights)
            info(f'done ({str(timedelta(seconds=time() - t0))} elapsed)')
        info(f'{inter_pair_nr=}, {intra_pair_nr=}')
        # cluster groups by system params
        if (len(pair_nrs) > 0):
            print(f'number of pairs per dataset ({len(pair_nrs)}): min={min(pair_nrs.values())}, max={max(pair_nrs.values())}')
        pair_nrs_precluster = pair_nrs.copy()
        pair_nrs_cluster_min = {}
        if (self.cluster):
            cluster_sys = {g: self.x_sys[x1_indices[group_index_start[g]]][:self.x_sys_global_num] for g in pair_nrs
                           if group_index_end[g] != group_index_start[g]} # empty group
            clusters = {}
            for g, sysf in cluster_sys.items():
                clusters.setdefault(tuple(sysf), []).append(g)
            pprint(clusters)
            clusters = list(clusters.values())
            pprint(pair_nrs)
            for c in clusters:
                pair_num_sum = sum([pair_nrs[g] for g in c])
                pair_num_min = min([pair_nrs[g] for g in c])
                for g in c:
                    pair_nrs[g] = pair_num_sum
                    pair_nrs_cluster_min[g] = pair_num_min
            if (len(pair_nrs) > 0):
                print(f'number of pairs per cluster ({len(clusters)}): min={min(pair_nrs.values())}, max={max(pair_nrs.values())}')
        nr_group_pairs_max = max(list(pair_nrs.values()) + [0])
        downsample_nr = min(list(pair_nrs.values()) + [np.infty]) * self.downsample_factor
        pprint(pair_nrs)
        info('computing pair weights')
        for g in pair_nrs:
            weight_modifier = self.confl_weight # confl pairs are already balanced by weight; here they can be boosted additionally
            if (self.downsample_groups):
                downsample_nr_g = int(np.ceil(downsample_nr / (pair_nrs[g] / pair_nrs_precluster[g])))
                actual_downsample_nr_g = min([downsample_nr_g, group_index_end[g] - group_index_start[g]])
                print(f'{g}: {actual_downsample_nr_g=} = {downsample_nr=} / ({pair_nrs[g]=} / {pair_nrs_precluster[g]=})'
                      + (f' [SHOULD BE {downsample_nr_g} ({actual_downsample_nr_g/downsample_nr_g:.0%})]'
                         if downsample_nr_g != actual_downsample_nr_g else ''))
                downsample_whitelist = set(sample(range(group_index_start[g], group_index_end[g]), actual_downsample_nr_g))
            # TODO: make sure many conflicting pairs are included in the sample
            for i in range(group_index_start[g], group_index_end[g]):
                if self.downsample_groups and i not in downsample_whitelist:
                    if self.downsample_always_confl and frozenset([self.x_ids[x1_indices[i]], self.x_ids[x2_indices[i]]]) in self.conflicting_smiles_pairs:
                        pass    # with this option, conflicting pairs are never removed in downsampling
                    else:
                        weights[i] = None
                        continue
                rt_diff = (np.infty if isinstance(g, tuple) # no statement can be made for inter-group pairs
                           or not self.use_pair_weights
                           else np.abs(self.y[x1_indices[i]] - self.y[x2_indices[i]]))
                if self.use_group_weights:
                    if self.group_weights_only_intra_cluster:
                        nr_group_pairs = pair_nrs_precluster[g]
                        nr_group_pairs_max = pair_nrs_cluster_min[g]
                    else:
                        nr_group_pairs = pair_nrs[g]
                        nr_group_pairs_max = nr_group_pairs_max
                else:
                    nr_group_pairs = nr_group_pairs_max
                weights_mod = pair_weights(smiles1=self.x_ids[x1_indices[i]], smiles2=self.x_ids[x2_indices[i]],
                                           rt_diff=rt_diff,
                                           nr_group_pairs=nr_group_pairs, nr_group_pairs_max=nr_group_pairs_max,
                                           confl_weights_modifier=weight_modifier, confl_pair_list=self.conflicting_smiles_pairs,
                                           only_confl=self.only_confl,
                                           weight_steepness=self.weight_steepness,
                                           weight_mid=self.weight_mid,
                                           max_rt=groups_max_rts[g] if self.dynamic_weights else None,
                                           epsilon=self.epsilon, discard_smaller_than_epsilon=self.discard_smaller_than_epsilon)
                if (rt_diff < self.epsilon and weights_mod is not None):
                    print(rt_diff, 'should this pair not have been discarded?')
                weights[i] = (weights_mod * weights[i]) if weights_mod is not None else None
        # NOTE: pair weights can be "None"
        info('done. removing None weights')
        # remove Nones
        x1_indices_new = []
        x2_indices_new = []
        y_trans_new = []
        weights_new = []
        is_confl_new = []
        removed_counter = 0
        for i in range (len(y_trans)):
            if (weights[i] is not None):
                x1_indices_new.append(x1_indices[i])
                x2_indices_new.append(x2_indices[i])
                y_trans_new.append(y_trans[i])
                weights_new.append(weights[i])
                is_confl_new.append(is_confl[i])
            else:
                removed_counter += 1
        info(f'removed {removed_counter} (of {len(y_trans)}) pairs for having "None" weights')
        info('done generating pairs')
        return dict(x1_indices= np.asarray(x1_indices_new),
                    x2_indices=np.asarray(x2_indices_new),
                    y_trans=np.asarray(y_trans_new),
                    weights=np.asarray(weights_new),
                    is_confl=np.asarray(is_confl_new))


    @staticmethod
    def weight_fn(x, steep=4, mid=0.75):
        """sigmoid function with f(0) → 0, f(2) → 1, f(0.75) = 0.5"""
        return 1 / (1 + np.exp(-steep * (x - mid)))

    @staticmethod
    def dataset_pair_it(indices, pair_step=1, pair_stop=None,
                        max_indices_size=None, max_num_pairs=None,
                        obl_indices=set()):
        n = len(indices)

        if (max_indices_size is not None):
            it = sorted(sample(list(range(n)), min(max_indices_size, n)))
        elif (max_num_pairs is not None):
            it = sample(list(range(n)), n)
        else:
            it = range(n)
        non_obl_pairs = 0
        do_break = False
        for i in it:
            if do_break:
                break
            for j in range(i + 1,
                           (n if pair_stop is None else min(i + pair_stop, n)),
                           pair_step):
                if (frozenset((indices[i], indices[j])) not in obl_indices):
                    if (max_num_pairs is not None and non_obl_pairs > max_num_pairs):
                        do_break = True
                        break
                    yield indices[i], indices[j], 1.0
                    non_obl_pairs += 1
        if (len(obl_indices) > 0):
            obl_weight = non_obl_pairs / len(obl_indices)
            print(f'{non_obl_pairs} non-conflicting pairs, {len(obl_indices)} conflicting pairs; weight: {obl_weight:.2f}')
            for i, j in obl_indices:
                yield i, j, obl_weight

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
        # pos: eluting second, neg: eluting first; (pos, neg) := 1  <->  (neg, pos) := 0(-1)
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


    def preprocess_doublets(self):
        doublet_rt_ranges = {}  # {(ds, id_): (1.2, 2.1)}
        for i in range(len(self.y)):
            rt = self.y[i]
            id_ = self.x_ids[i]
            ds = self.dataset_info[i]
            if ((ds, id_) not in doublet_rt_ranges):
                doublet_rt_ranges[(ds, id_)] = (rt, rt)
            doublet_rt_ranges[(ds, id_)] = (min(rt, *doublet_rt_ranges[(ds, id_)]),
                                            max(rt, *doublet_rt_ranges[(ds, id_)]))
        self.doublet_rt_ranges = {k: v for k, v in doublet_rt_ranges.items()
                                  if v[0] != v[1]}
        # stats on doublets: how many per dataset? mean/median rt difference per doublet
        data = pd.DataFrame.from_records([{'dataset': k[0], 'rt_diff': v[1] - v[0]}
                                          for k, v in self.doublet_rt_ranges.items()])
        if len(data) > 0:
            stats = data.groupby('dataset').rt_diff.agg(['count', 'mean', 'median'])
            print('doublet stats:\n' + stats.to_string())



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

    def remove_indices(self, indices):
        assert all(len(_) == len(self.x1_indices) for _ in [
            self.x1_indices, self.x2_indices, self.y_trans, self.weights,
            self.is_confl])
        x1_indices_new = []
        x2_indices_new = []
        y_trans_new = []
        weights_new = []
        is_confl_new = []
        indices = set(indices)
        for i in range(len(self.x1_indices)):
            if (i not in indices):
                x1_indices_new.append(self.x1_indices[i])
                x2_indices_new.append(self.x2_indices[i])
                y_trans_new.append(self.y_trans[i])
                weights_new.append(self.weights[i])
                is_confl_new.append(self.is_confl[i])
        self.x1_indices = np.asarray(x1_indices_new)
        self.x2_indices = np.asarray(x2_indices_new)
        self.y_trans = np.asarray(y_trans_new)
        self.weights = np.asarray(weights_new)
        self.is_confl = np.asarray(is_confl_new)

    def __len__(self):
        return self.y_trans.shape[0]

    def __getitem__(self, index):
        # x1_sys == x2_sys for the first `x_info_global_num` features
        # returns ((graph, extra, sys) x 2, y, weight)
        return (((self.x_mols[self.x1_indices[index]], self.x_extra[self.x1_indices[index]],
                  self.x_sys[self.x1_indices[index]]),
                 (self.x_mols[self.x2_indices[index]], self.x_extra[self.x2_indices[index]],
                  self.x_sys[self.x2_indices[index]])),
                self.y_trans[index], self.weights[index], self.is_confl[index])

def check_integrity(x: RankDataset, clean=False):
    pairs = {}
    for i, (x1, x2, y) in enumerate(zip(x.x1_indices, x.x2_indices, x.y_trans)):
        p = tuple(sorted([x.x_ids[x1], x.x_ids[x2]]))
        if (p[0] == x.x_ids[x2]):
            y = (-1 if x.y_neg else 0) if y == 1 else 1
        pairs.setdefault(p, []).append(
            (x.dataset_info[x1], y, x.x_sys[x1][:x.x_sys_global_num], i))
        # NOTE: only taking the global sys features makes most sense, although due to different
        # gradient positions, pairs cleaned in this manner *technically can be possible*.
    records = []
    clean_indices = []
    same_settings_datasets = []
    for v in pairs.values():
        nr_confl = nr_invalid = nr_combs = 0
        invalid = []
        for (ds_i, y_i, sys_i, i), (ds_j, y_j, sys_j, j) in combinations(v, 2):
            nr_combs += 1
            if (y_i != y_j):
                if (ds_i == ds_j):
                    print(ds_i, x.x_ids[x.x1_indices[i]], x.x_ids[x.x2_indices[i]],
                          x.x_ids[x.x1_indices[j]], x.x_ids[x.x2_indices[j]])
                nr_confl += 1
                if ((sys_i == sys_j).all()):
                    nr_invalid += 1
                    same_settings_datasets.append((ds_i, ds_j))
                    invalid.append((i, j))
        if (clean):
            # gready algorithm to remove indices with most invalid pairs
            while (len(invalid) > 0):
                max_i = Counter(np.asarray(invalid).flatten()).most_common()[0][0]
                clean_indices.append(max_i)
                invalid = [_ for _ in invalid if max_i not in _]
        records.append(dict(nr_combs=nr_combs, nr_confl=nr_confl, nr_invalid=nr_invalid))
    stats = pd.DataFrame.from_records(records)
    if (len(stats) != 0):
        print(f'conflicting pairs percentage: {stats.nr_confl.sum() / stats.nr_combs.sum():.2%}')
        print(f'conflicting pairs percentage (averaged): {(stats.nr_confl / stats.nr_combs).mean():.2%}')
        print(f'invalid conflicting pairs percentage: {stats.nr_invalid.sum() / stats.nr_confl.sum():.2%}')
        print(f'invalid pairs percentage (of total): {stats.nr_invalid.sum() / stats.nr_combs.sum():.2%}')
    # dss = pd.merge(dss, pd.read_csv(os.path.join('/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/', 'ph_info.csv'), sep='\t', index_col=0)[REL_ONEHOT_COLUMNS], how='left', left_index=True, right_index=True)
    # for ds1, ds2 in same_settings_datasets:
    #     assert all(x[0][0] == x[0][1] or np.isnan(x[0][0]) and np.isnan(x[0][1]) for x in
    #                zip(dss.loc[[ds1, ds2], ['column.id', 'column.flowrate', 'column.length'] +
    #                    ['class.pH.A', 'class.pH.B', 'class.solvent'] + ['H']].values.transpose()))
    return stats, clean_indices, same_settings_datasets
