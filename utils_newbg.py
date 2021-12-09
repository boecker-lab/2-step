from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union
from chemprop.features.featurization import BatchMolGraph
from torch.utils.data import Dataset
import numpy as np
import logging
from logging import info, warning
from time import time
from datetime import timedelta
from itertools import combinations, product
from random import sample, shuffle
from utils import pair_weights

logger = logging.getLogger('rtranknet.utils')
info = logger.info
warning = logger.warning


@dataclass
class RankDataset(Dataset):
    x_mols: List[Union[BatchMolGraph, str]]       # smiles or mol graphs
    x_extra: Union[np.ndarray, List[List[float]]] # extra compound features, e.g., logp
    x_sys: Union[np.ndarray, List[List[float]]]   # system features
    x_ids: List[str]                              # ID (e.g., smiles) for each sample
    y: Union[np.ndarray, List[float]]             # retention times
    use_pair_weights: bool=True                   # use pair weights (epsilon)
    epsilon: float=0.5                                  # soft threshold for retention time difference
    use_group_weights: bool=True                  # weigh number of samples per group
    cluster: bool=False                           # cluster datasets with same column params for calculating
                                                  # group weights
    weight_steepness: float=4                     # steepness of the pair_weight_fn
    weight_mid: float=0.75                        # mid-factor of the weight_mid
    pair_step: int=1                              # step size for generating pairs
    pair_stop: Optional[int]=None                 # stop number for generating pairs
    dataset_info: Optional[List[str]] = None      # Dataset ID for each datum
    void_info: Optional[Dict[str, float]] = None  # void time mapping for dataset IDs
    void: Optional[float]=None                    # global void time
    no_inter_pairs: bool=True                     # don't generate inter dataset pairs
    no_intra_pairs: bool=False                    # don't generate intra dataset pairs
    max_indices_size:Optional[int]=None           # limit for the size of indices
    y_neg : bool=False                            # -1 instead of 0 for negative pair
    conflicting_smiles_pairs:dict = field(default_factory=dict) # conflicting pairs (smiles)
    only_confl: bool=False                                          # gather only conflicting pairs
    confl_weight: float=100.                                        # weight modifier for conflicting pairs

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
        # transform single compounds(+info) into pairs for ranking
        (self.x1_indices, self.x2_indices, self.y_trans,
         self.weights, self.sys_indices, self.is_confl) = self._transform_pairwise()

    def _transform_pairwise(self):
        x1_indices = []
        x2_indices = []
        y_trans = []
        weights = []
        sys_indices = []
        is_confl = []
        # group by dataset
        groups = {}
        pair_nrs = {}
        group_index_start = {}
        group_index_end = {}
        # confl_pair_report = {}
        if (self.dataset_info is None):
            groups['unk'] = list(range(len(self.y)))
        else:
            for i in range(len(self.y)):
                groups.setdefault(self.dataset_info[i], []).append(i)
        # preprocess confl pair list for O(1) lookup
        # and disregard confl pairs not conflicting for this training set
        confl_pairs_lookup = {k for k, v in self.conflicting_smiles_pairs.items()
                              if any(x[0] in groups and x[1] in groups
                                     for x in v)}
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
                                          obl_indices=confl_indices)
                if (logger.level <= logging.INFO):
                    from tqdm import tqdm
                    it = tqdm(it)
                for i, j, w in it:
                    res = self.get_pair(self.y, i, j, group_void_rt or 0, group_void_rt or 0, self.y_neg)
                    if (res is None):
                        continue
                    pos_idx, neg_idx, yi = res
                    x1_indices.append(pos_idx)
                    x2_indices.append(neg_idx)
                    y_trans.append(yi)
                    # weights
                    weights.append(w)
                    # sysinfo
                    sys_indices.append(pos_idx)
                    # is conflicting pair?
                    is_confl.append(frozenset((pos_idx, neg_idx)) in confl_indices)
                    pair_nr += 1
                pair_nrs[group] = pair_nr
                intra_pair_nr += pair_nr
                group_index_end[group] = len(weights)
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
                    # NOTE: sysinfo does not work for inter pairs, therefore append None to get runtime error
                    sys_indices.append(None)
                    is_confl.append(None)
                    pair_nr += 1
                pair_nrs[(group1, group2)] = pair_nr
                inter_pair_nr += pair_nr
                group_index_end[(group1, group2)] = len(weights)
            info(f'done ({str(timedelta(seconds=time() - t0))} elapsed)')
        info(f'{inter_pair_nr=}, {intra_pair_nr=}')
        # cluster groups by system params
        if (self.cluster):
            cluster_sys = {g: self.x_sys[sys_indices[group_index_start[g]]] for g in pair_nrs
                           if group_index_end[g] != group_index_start[g]} # empty group
            clusters = {}
            for g, sysf in cluster_sys.items():
                clusters.setdefault(tuple(sysf), []).append(g)
            from pprint import pprint
            pprint(clusters)
            clusters = list(clusters.values())
            pprint(pair_nrs)
            for c in clusters:
                pair_num_sum = sum([pair_nrs[g] for g in c])
                for g in c:
                    pair_nrs[g] = pair_num_sum
            pprint(pair_nrs)
        all_groups_list = list(pair_nrs)
        nr_group_pairs_max = max(list(pair_nrs.values()) + [0])
        info('computing pair weights')
        for g in pair_nrs:
            weight_modifier = self.confl_weight # confl pairs are already balanced by weight; here they can be boosted additionally
            for i in range(group_index_start[g], group_index_end[g]):
                rt_diff = (np.infty if isinstance(g, tuple) # no statement can be made for inter-group pairs
                           or not self.use_pair_weights
                           else np.abs(self.y[x1_indices[i]] - self.y[x2_indices[i]]))
                weights_mod = pair_weights(self.x_ids[x1_indices[i]], self.x_ids[x2_indices[i]], rt_diff,
                                           pair_nrs[g] if self.use_group_weights else nr_group_pairs_max,
                                           nr_group_pairs_max, weight_modifier, self.conflicting_smiles_pairs,
                                           only_confl=self.only_confl)
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
        return np.asarray(x1_indices_new), np.asarray(x2_indices_new), np.asarray(
            y_trans_new), np.asarray(weights_new), np.asarray(sys_indices), np.asarray(is_confl)


    @staticmethod
    def weight_fn(x, steep=4, mid=0.75):
        """sigmoid function with f(0) → 0, f(2) → 1, f(0.75) = 0.5"""
        return 1 / (1 + np.exp(-steep * (x - mid)))

    @staticmethod
    def dataset_pair_it(indices, pair_step=1, pair_stop=None,
                        max_indices_size=None, obl_indices=set()):
        n = len(indices)
        if (max_indices_size is None):
            it = range(n)
        else:
            it = sorted(sample(list(range(n)), min(max_indices_size, n)))
        non_obl_pairs = 0
        for i in it:
            for j in range(i + 1,
                           (n if pair_stop is None else min(i + pair_stop, n)),
                           pair_step):
                if (frozenset((indices[i], indices[j])) not in obl_indices):
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

    def __len__(self):
        return self.y_trans.shape[0]

    def __getitem__(self, index):
        # for now, only pairs from one system are returned; thus x1_sys == x2_sys
        # returns ((graph, extra, sys) x 2, y, weight)
        return (((self.x_mols[self.x1_indices[index]], self.x_extra[self.x1_indices[index]],
                  self.x_sys[self.sys_indices[index]]),
                 (self.x_mols[self.x2_indices[index]], self.x_extra[self.x2_indices[index]],
                  self.x_sys[self.sys_indices[index]])),
                self.y_trans[index], self.weights[index], self.is_confl[index])
