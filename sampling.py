from typing import Literal
from utils_newbg import RankDataset
import pickle
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import numpy as np
import torch

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.cpu().numpy() / torch.sum(self.weights).cpu().numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def calc_sampling_weights(td: RankDataset, method: Literal['compounds', 'pairs'],
                          cluster_informed=False, sqrt_weights=False, verbose=False):
    sets = pd.DataFrame(dict(sets=[td.dataset_info[i] for i in td.x1_indices]))
    if (cluster_informed):
        clusters = {ds: [i for i, c in enumerate(td.dataset_clusters) if ds in c][0]
                    for ds in [ds for c in td.dataset_clusters for ds in c]}
        sets['cluster'] = sets.sets.map(clusters.__getitem__)
        grouping = 'cluster'
    else:
        grouping = 'sets'
    if (method == 'pairs'):
        counts = sets[grouping].value_counts()
    else:
        compounds = pd.DataFrame(dict(sets=td.dataset_info))
        if (cluster_informed):
            compounds['cluster'] = compounds.sets.map(clusters.__getitem__)
        counts = compounds[grouping].value_counts()
    sets['counts'] = sets[grouping].map(counts)
    if (sqrt_weights):
        sets['counts'] = np.sqrt(sets['counts'])
    sets['weights'] = 1 / (sets.counts / sets.counts.min())
    if (verbose):
        print(f'based on {method}' + (', using clusters' if cluster_informed else '')
              + (', using sqrt' if sqrt_weights else '')
              + f': {sets.weights.agg(["min", "max", "mean", "median"])}')
    return sets.weights.values
