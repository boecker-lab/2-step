import tensorflow as tf
from itertools import combinations, product
from random import random, sample, shuffle
from sklearn.utils import class_weight
from scipy.sparse import  issparse
import sys
from pprint import pformat

def csr2tf(csr):
    indices = []
    values = []
    for (i, j), v in csr.todok().items():
        indices.append([i, j])
        values.append(v)
    return tf.sparse.SparseTensor(indices, values, csr.shape)

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
