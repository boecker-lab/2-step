from itertools import combinations
from logging import basicConfig, INFO, info, warning
import pandas as pd
import numpy as np
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import MolFromSmiles
import os
import pickle
import json
import re
from tap import Tap
from typing import List, Optional, Literal, Tuple, Union
from tqdm import tqdm
import pickle
import io
import bisect
from collections import Counter

import torch

from utils import REL_COLUMNS, Data, export_predictions
from features import features, parse_feature_spec

BENCHMARK_DATASETS = ['0003', '0010', '0018', '0055', '0054', '0019', '0002']

def get_authors(ds, repo_root_dir='/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/'):
    return str(pd.read_csv(repo_root_dir + f'/processed_data/{ds}/{ds}_info.tsv', sep='\t')['authors'].item())

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def eval_(y, preds, epsilon=0.5, void_rt=0.0, roi_thr=1e-5, dont_count_low_epsilon=False):
    assert len(y) == len(preds)
    if (not any(preds)):
        return 0.0
    preds, y = zip(*sorted(zip(preds, y)))
    matches = 0
    total = 0
    for i, j in combinations(range(len(y)), 2):
        if y[i] <= void_rt and y[j] <= void_rt:
            # ignore pairs with both compounds eluting in void volume
            continue
        if dont_count_low_epsilon and np.abs(y[i] - y[j]) < epsilon:
            continue
        diff = y[i] - y[j]
        #if (diff < epsilon and preds[i] < preds[j]):
        if (diff < epsilon and (preds[j] - preds[i] > roi_thr)):
            matches += 1
        total += 1
    toret = matches / total if not total == 0 else np.nan
    return toret

def eval_from_pairs(y, pair_preds, allow_0_preds=False, epsilon=0.5, void_rt=0.0):
    assert pair_preds.shape == (len(y), len(y))
    assert np.allclose(pair_preds, -pair_preds.T), 'not symmetrical'
    matches = total = 0
    for i, j in combinations(range(len(y)), 2):
        if (np.abs(y[i] - y[j]) < epsilon):
            continue
        if y[i] <= void_rt and y[j] <= void_rt:
            continue
        if np.isclose(pair_preds[i, j], 0):
             if allow_0_preds:
                 continue
             else:
                 total += 1     # wrong prediction
                 continue
                 # raise Exception('predictions are not all 1s and -1s:', pair_preds[i, j])
        if pair_preds[i, j] * (y[j] - y[i]) > 0:
            matches += 1
        total += 1
    toret = matches / total if not total == 0 else np.nan
    return toret

def order_from_pairs(pair_preds):
    from graphlib import TopologicalSorter
    assert np.allclose(pair_preds, -pair_preds.T), 'not symmetrical'
    ts = TopologicalSorter()
    for i, j in combinations(range(len(pair_preds)), 2):
        if np.isclose(pair_preds[i, j], 1):
            ts.add(j, i)
        elif np.isclose(pair_preds[i, j], -1):
            ts.add(i, j)
    return tuple(ts.static_order())

def eval_detailed(mols, y, preds, epsilon=0.5, void_rt=0.0, roi_thr=1e-5):
    matches = []
    assert len(y) == len(preds)
    preds, y, mols = zip(*sorted(zip(preds, y, mols)))
    total = 0
    if (any(preds)):
        for i, j in combinations(range(len(y)), 2):
            if y[i] <= void_rt and y[j] <= void_rt:
                # ignore pairs with both compounds eluting in void volume
                continue
            diff = y[i] - y[j]
            roi_diff = preds[j] - preds[i]
            if (diff < epsilon and (roi_diff > roi_thr)):
                matches.append((frozenset([mols[i], mols[j]]), roi_diff))
            total += 1
        return len(matches) / total if not total == 0 else np.nan, matches
    else:
        return 0.0, []

def eval2(df, epsilon=0.5, classyfire_level=None):
    df_eval = df.dropna(subset=['rt', 'roi'])
    df_eval.reset_index(drop=True, inplace=True)
    classes = (list(set(df_eval[classyfire_level].dropna().tolist()))
               if classyfire_level is not None else []) + ['total']
    matches = {c: [0 for i in range(len(df_eval))] for c in classes}
    total = {c: 0 for c in classes}
    for i, j in combinations(range(len(df_eval)), 2):
        rt_diff = df_eval.rt[i] - df_eval.rt[j]
        for c in classes:
            if (c != 'total' and
                df_eval[classyfire_level][i] == c or df_eval[classyfire_level][j] == c):
                match = 0
            else:
                match = ((np.sign(rt_diff) == np.sign(df_eval.roi[i] - df_eval.roi[j]))
                         or (np.abs(rt_diff) < epsilon)).astype(int)
                total[c] += 2
            matches[c][i] += match
            matches[c][j] += match
    df_eval['matches'] = matches['total']
    df_eval['matches_perc'] = df_eval.matches / total['total']
    df_classes = pd.DataFrame({'matches': matches})
    return (df_eval.matches.sum() / total['total'], df_eval,
            {'matches': {c: np.sum(matches[c]) for c in classes},
             'matches_perc': {c: np.sum(matches[c]) / total[c] for c in classes}})

def lcs(seq1, seq2):
    m = len(seq1)
    n = len(seq2)
    LCS = np.full((m + 1, n + 1), np.nan)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                LCS[i, j] = 0
            elif seq1[i-1] == seq2[j-1]:
                LCS[i, j] = LCS[i-1, j-1]+1
            else:
                LCS[i, j] = max(LCS[i-1, j], LCS[i, j-1])
    return LCS[m][n]

def lis_prepare(seq1, seq2):
    # delete doublets
    seq1_occs = Counter(seq1)
    seq2_occs = Counter(seq2)
    seq1_cleaned = [c for c in seq1 if seq1_occs[c] == 1]
    seq2_cleaned = [c for c in seq2 if seq2_occs[c] == 1]
    shared_compounds = set(seq1_cleaned) & set(seq2_cleaned)
    seq1_shared = [c for c in seq1_cleaned if c in shared_compounds]
    seq2_shared = [c for c in seq2_cleaned if c in shared_compounds]
    mapping = {c: i for i, c in enumerate(seq1_shared)}
    return [mapping[c] for c in seq2_shared]

def lis_len(seq):                   # from https://algorithmist.com/wiki/Longest_increasing_subsequence, O(Nlogk)
    # the smallest number that ends a chain of length i
    best = []
    for num in seq:
        # find the smallest index where num is bigger than best[i]
        i = bisect.bisect_left(best, num)
        # if num is bigger than longest increasing subsequence
        # so far, we create a new length
        if i >= len(best):
            best.append(num)
        else:
            # update because by definition, num is smaller best[i]
            best[i] = num
    return len(best)

def lis(seq1, seq2):
    return lis_len(lis_prepare(seq1, seq2))

def lcs_results(df, mode='lis'):
    order_true = df.sort_values('rt').smiles.tolist()
    order_pred = df.sort_values('roi').smiles.tolist()
    lcs_fun = {'lcs': lcs, 'lis':lis}[mode]
    return lcs_fun(order_true, order_pred)

def rt_roi_diffs(data, y, preds, k=3):
    """for all pairs x, y:
    is |rt_x - rt_y| very different from |roi_x - roi_y|?
    - increment outl[x], outl[y]
    - at the end return k u_i's with highest outl[u_i]
    """
    from pygam import LinearGAM
    assert len(y) == len(preds)
    scale_roi = max(preds) - min(preds)
    scale_rt = max(y) - min(y)
    df = pd.DataFrame(data.df.iloc[np.concatenate((data.train_indices, data.test_indices, data.val_indices))])
    df['roi'] = preds
    df.dropna(subset=['roi', 'rt'], inplace=True)
    df.sort_values(by='rt', inplace=True)
    # diffs = np.zeros((len(df)))
    # for i, j in combinations(range(len(y)), 2):
    #     diff_roi = np.abs(preds[i] - preds[j]) * scale_roi
    #     diff_rt = np.abs(y[i] - y[j]) * scale_rt
    #     diffs[i] += np.abs(diff_roi - diff_rt) / (len(y) ** 2)
    #     diffs[j] += np.abs(diff_roi - diff_rt) / (len(y) ** 2)
    # for i in range(k, len(df) - k):
    #     window = np.concatenate((df.roi[i-k:i], df.roi[i+1:i+k+1]))
    #     roi_mean = np.mean(window)
    #     diffs[i] = np.abs(df.roi[i] - roi_mean)
    gam = LinearGAM().fit(df.rt, df.roi)
    df['diffs'] = np.abs(df.roi - gam.predict(df.rt))
    df['rt_gam'] = LinearGAM().fit(df.roi, df.rt).predict(df.roi)
    df['diffs'] = (df['diffs'] > 0.2 * (np.sum(np.abs([min(df.roi), max(df.roi)])))).astype(int)
    return df

def visualize_df(df, x_axis='rt'):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = ('rt', 'roi') if x_axis == 'rt' else ('roi', 'rt')
    points = ax.scatter(df[x], df[y], c=df.diffs, cmap='coolwarm')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(df.id[0].split('_')[0])

    # blank image
    imm = Image.new('RGBA', (300, 300))
    im = OffsetImage(np.array(imm), zoom=0.5)
    xybox=(100., 100.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def recolor(df, ind, points, k=5):
        # find k points with closest rt
        rts = df.rt.values
        rt = rts[ind]
        rt_inds = np.argsort(np.abs(rts - rt))[:k]
        # find k points with closest roi
        rois = df.roi.values
        roi = rois[ind]
        roi_inds = np.argsort(np.abs(rois - roi))[:k]
        cols = {(True, False): [0, 0, 1, 1],
                (False, True): [1, 1, 0, 1],
                (True, True): [0, 1, 0, 1],
                (False, False): [0, 0, 0, 1]}
        colors = [cols[(p in rt_inds, p in roi_inds)] for p in range(len(rts))]
        return colors


    def hover(event):
        if (not hasattr(points, 'def_colors')):
            points.def_colors = points.get_facecolors()
        # if the mouse is over the scatter points
        if points.contains(event)[0]:
            # find out the index within the array from the event
            ind = points.contains(event)[1]["ind"][0]
            points.set_facecolors(recolor(df, ind, points))
            # get the figure size
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy =(df.rt.iloc[ind], df.roi.iloc[ind])
            # set the image corresponding to that point
            im.set_data(np.array(MolToImage(MolFromSmiles(df.smiles.iloc[ind]), (300, 300))))
        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
            points.set_facecolors(points.def_colors)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()


def data_stats(d, data, custom_column_fields=None, validation_counts_as_train=False, compound_identifier='smiles'):
    ds = d.df.dataset_id.unique().item()
    train_df = data.df.loc[data.df.split_type.isin(
        ['train'] + (['val'] if validation_counts_as_train else []))]
    train_compounds_all = set(train_df[compound_identifier])
    this_column = d.df['column.name'].unique().item()
    train_compounds_col = set(train_df.loc[train_df['column.name'] == this_column, compound_identifier])
    test_compounds = set(d.df[compound_identifier])
    system_fields = custom_column_fields + ['ph'] if data.use_ph else []
    train_configs = [t[1:] for t in set(train_df[['dataset_id', 'column.name'] + system_fields]
                                        .itertuples(index=False, name=None))]
    test_config = tuple(d.df[['column.name'] + system_fields].iloc[0].tolist())
    same_config = len([t for t in train_configs if t == test_config])
    same_column = len([t for t in train_configs if t[0] == test_config[0]])
    stats = {'num_data': len(test_compounds),
            'compound_overlap_all': (len(test_compounds & train_compounds_all)
                                           / len(test_compounds)),
            'compound_overlap_column': (len(test_compounds & train_compounds_col)
                                              / len(test_compounds)),
            'column_occurences': same_column,
            'config_occurences': same_config}
    flags = {
        'our_datasets': 'harrieder' in get_authors(ds, d.repo_root_folder).lower(),
        'benchmark_dataset': ds in BENCHMARK_DATASETS,
        'structure_disjoint': stats['compound_overlap_all'] == 0,
        'setup_disjoint': stats['config_occurences'] == 0,
        'column_disjoint': stats['column_occurences'] == 0
    }
    stats['flags'] = flags
    return stats




def predict(X, model, batch_size):
    from keras import backend as K
    preds = []
    ranker_output = K.function([model.layers[0].input], [model.layers[-3].get_output_at(0)])
    for x in np.array_split(X, np.ceil(X.shape[0] / batch_size * 10)):
        preds.append(ranker_output([x])[0].ravel())
    return np.concatenate(preds)

class EvalArgs(Tap):
    model: str # model to load
    test_sets: List[str] # either CSV or dataset IDs to evaluate on
    model_type: Literal['ranknet', 'mpn', 'rankformer_rt', 'rankformer'] = 'mpn'
    gpu: bool = False
    batch_size: int = 256
    no_isomeric: bool = False
    repo_root_folder: str = '/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/' # location of the dataset github repository
    add_desc_file: str = '/home/fleming/Documents/Projects/rtranknet/data/qm_merged.csv' # csv with additional features with smiles as identifier
    output: Optional[str] = None # write output to json file
    verbose: bool = False
    no_progbar: bool = False # no progress-bar
    void_rt: float = 0.0
    no_metadata_void_rt: bool = False # don't use t0 value from repo metadata (times 2)
    remove_void_compounds: bool = False      # remove void compounds completely
    include_void_compounds_mcd: bool = False # don't remove compounds eluting in void volume for minimum compound deletion
    void_factor: float = 2              # factor for 'column.t0' value to use as void threshold
    cache_file: str = 'cached_descs.pkl'
    export_rois: bool = False
    export_rois_dir: Optional[str] = None
    export_embeddings: bool = False
    device: Optional[str] = None # can be `mirrored`, a specific device name like `gpu:1` or `None` which automatically selects an option
    epsilon: Union[str, float] = '30s' # difference in evaluation measure below which to ignore falsely predicted pairs
    # dont_count_low_epsilon: bool = False # completely ignore pairs with diff < epsilon for acc calculation
    remove_train_compounds: bool = False
    remove_train_compounds_mode: Literal['all', 'column', 'print'] = 'all'
    compound_identifier: Literal['smiles', 'inchi.std', 'inchikey.std'] = 'smiles' # how to identify compounds for statistics
    plot_diffs: bool = False    # plot for every dataset with outliers marked
    test_stats: bool = False    # overview stats for all datasets
    dataset_stats: bool = False # stats for each dataset
    no_optional_stats: bool = False   # don't do optional stats for conflicting pairs
    diffs: bool = False         # compute outliers
    classyfire: bool = False    # compound class stats
    confl_pairs: Optional[str] = None # pickle file with conflicting pairs (smiles)
    overwrite_system_features: List[str] = [] # use these system descriptors for confl pairs stats instead of those from the training data
    preds_from_exported_rois: List[str] = []
    get_more_dataset_info: bool = False # attempt to get more info from RepoRT on the datasets for more detailed stats
    mcd_method: Literal['lcs', 'lis'] = 'lis' # how to compute minimum compound deletion

    def process_args(self):
        # process epsilon unit
        self.epsilon = str(self.epsilon)
        if (match_ := re.match(r'[\d\.]+ *(min|s)', self.epsilon)):
            unit = match_.groups()[0]
            if unit == 's':
                self.epsilon = float(self.epsilon.replace('s', '').strip()) / 60
            elif unit == 'min':
                self.epsilon = float(self.epsilon.replace('min', '').strip())
            else:
                raise ValueError(f'wrong unit for epsilon ({self.epsilon}): {unit}')
        elif (re.match(r'[\d\.]+', self.epsilon)):
            self.epsilon = float(self.epsilon.strip())
        else:
            raise ValueError(f'wrong format for epsilon ({self.epsilon})')

    def configure(self) -> None:
        self.add_argument('--epsilon', type=str)

class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes' and not torch.cuda.is_available():
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_model(path: str, type_='mpn'):
    if (type_ == 'keras'):
        # NOTE: might be broken because of missing .tf, but not used anyways anymore
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        data = pickle.load(open(os.path.join(path, 'assets', 'data.pkl'), 'rb'))
        config = json.load(open(os.path.join(path, 'assets', 'config.json')))
    else:
        path = path + '.pt' if not path.endswith('pt') else path
        if (torch.cuda.is_available()):
            model = torch.load(path)
        else:
            model = torch.load(path, map_location=torch.device('cpu'))
            if hasattr(model, 'ranknet_encoder'):
                model.ranknet_encoder.embedding.gnn.device = torch.device('cpu')
                model.ranknet_encoder.embedding.gnn.encoder[0].device = torch.device('cpu')
            try:
                model.encoder.encoder[0].device = torch.device('cpu')
            except:
                pass
        path = re.sub(r'_ep\d+(\.pt)?$', '', re.sub(r'\.pt$', '', path)) # for ep_save
        data = DataUnpickler(open(f'{path}_data.pkl', 'rb')).load()
        config = json.load(open(f'{path}_config.json'))
    return model, data, config


def classyfire_stats(d: Data, args: EvalArgs, plot=False, compound_identifier='smiles'):
    acc2, results, matches = eval2(d.df, args.epsilon, 'classyfire.class')
    print(f'{ds}: {acc2:.2%} accuracy)')
    groups = results.groupby('classyfire.class')
    results['matches_perc'] = results.matches_perc * len(results)
    # print(groups.matches_perc.agg(['mean', 'median', 'std', 'count']))
    # print(results.groupby('classyfire.class').matches_perc.agg(['mean', 'median', 'std', 'count']))
    matches_df = pd.DataFrame.from_dict(matches['matches_perc'], orient='index', columns=['acc_without'])
    matches_df['acc_without_diff'] = matches_df.acc_without - acc2
    matches_df['num_compounds'] = ([len(d.df.loc[d.df['classyfire.class'] == c])
                                    for c in matches_df.index.tolist()[:-1]]
                                   + [len(d.df)])
    matches_df['class_perc'] = matches_df.num_compounds / len(d.df)
    train_compounds = []
    train_compounds_all = len(set(data.df[compound_identifier].tolist()))
    for c in matches_df.index.tolist()[:-1]:
        compounds_perc = len(set(data.df.loc[data.df['classyfire.class'] == c,
                                             compound_identifier].tolist())) / train_compounds_all
        train_compounds.append(compounds_perc)
    matches_df['class_perc_train'] = train_compounds + [1.0]
    matches_df.index = [re.sub(r' \(CHEMONTID:\d+\)', '', i) for i in matches_df.index]
    print(matches_df.sort_values(by='acc_without_diff', ascending=False)[
        ['acc_without_diff', 'num_compounds', 'class_perc', 'class_perc_train']])
    if (plot):       # plotting
        matches_df.drop('total').sort_values(by='acc_without_diff', ascending=False)[
            ['acc_without_diff', 'class_perc', 'class_perc_train']].plot(rot=20)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.show()

def compound_acc(y, preds, comp_index, epsilon=0.5):
    matches = 0
    total = 0
    for j in range(len(y)):
        diff = (y[comp_index] - y[j]) * - np.sign(preds[comp_index] - preds[j])
        if (diff < epsilon):
            matches += 1
        total += 1
    return matches / total

def compound_stats(d: Data, args:EvalArgs):
    # logp
    from rdkit.Chem.Descriptors import MolLogP
    d.df['MolLogP'] = [MolLogP(MolFromSmiles(s)) for s in d.df.smiles]
    # mean compound acc
    d.df['mean_acc'] = [compound_acc(d.df.rt.tolist(), d.df.roi.tolist(), i, epsilon=args.epsilon)
                        for i in range(len(d.df))]

def density_plot(df: pd.DataFrame, x, y):
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
    toplot = df.sort_values(by=x)[[x, y]].rolling(len(df) / 100).mean().dropna()
    xy = np.vstack([toplot[x], toplot[y]])
    z = gaussian_kde(xy)(xy)
    toplot.plot.scatter(x, y, c=z)
    plt.show()

def pair_stats(d: Data, verbose=False):
    fields = {}
    it = combinations(range(len(d.df)), 2)
    if (verbose):
        it = tqdm(it)
    for i, j in it:
        row_i, row_j = d.df.iloc[i], d.df.iloc[j]
        fields.setdefault('indices', []).append((i, j))
        fields.setdefault('abs_rt_diff', []).append(np.abs(row_i.rt - row_j.rt))
        fields.setdefault('abs_roi_diff', []).append(np.abs(row_i.roi - row_j.roi))
        fields.setdefault('prediction_correct', []).append(np.sign(row_i.rt - row_j.rt) == np.sign(row_i.roi - row_j.roi))
        if ('MolLogP' in d.df.columns):
            fields.setdefault('MolLogP_diff', []).append(np.abs(row_i.MolLogP - row_j.MolLogP))
    return pd.DataFrame(fields)

def get_pair_consensus_order(pair, train_df, epsilon=0.5):
    s1, s2 = sorted(pair)
    confl_df = train_df.loc[train_df.smiles.isin(pair)]
    counts = confl_df.dataset_id.value_counts()
    orders = []
    for ds in counts[counts > 1].index.tolist():
        ds_df = confl_df.loc[confl_df.dataset_id == ds]
        if not (len(ds_df) == 2 and ds_df.smiles.nunique() == 2):
            continue    # doublet
        order = ds_df.loc[ds_df.smiles == s1, 'rt'].item() - ds_df.loc[ds_df.smiles == s2, 'rt'].item()
        if np.abs(order) < epsilon:
            continue
        orders.append(order > 0)
    if len(orders) > 0:
        return np.mean(orders)
    return None

def get_pair_order(s1, s2, ds, rts, void, epsilon=0.5):
    try:
        rt1 = rts[ds][s1]
        rt2 = rts[ds][s2]
    except:
        # doublets etc.
        return None
    if (np.abs(rt1 - rt2) < epsilon or (rt1 < void and rt2 < void)):
        return None
    return rt1 - rt2 > 0

def get_pair_stats(df, ds_target, qualifiers, confl_pairs, void_info, epsilon=0.5):
    rts = {}
    rel_datasets = set(df.dataset_id.unique().tolist())
    for ds in rel_datasets:
        rows = df.loc[(df.dataset_id == ds)].drop_duplicates('smiles.std', keep=False)
        for smiles, rt in rows[['smiles.std', 'rt']].dropna().values:
            rts.setdefault(ds, {})[smiles] = rt
    setups = df.drop_duplicates(subset=['dataset_id']).groupby(qualifiers)['dataset_id'].agg(list)
    setups_strict = df.drop_duplicates(subset=['dataset_id']).groupby([c for c in df.columns.tolist() if c.startswith('column.')]
                                                                      + ['mobilephase', 'gradient', 'authors'], dropna=False)['dataset_id'].agg(list)
    setup_names = setups.index.tolist()
    setup_sets = list(setups)
    setup_dict = {k: set(v) for k, v in dict(setups).items()}
    setup_dict_rev = {}
    setups_strict_lookup = {}
    for cluster in list(setups_strict):
        for ds in cluster:
            setups_strict_lookup[ds] = {x for x in cluster if x != ds}
    for setup, sets in setup_dict.items():
        for ds in sets:
            setup_dict_rev[ds] = setup
    records = []
    no_setup = []
    if ds_target not in setup_dict_rev:
        # can't make statistics
        return None
    for p in tqdm(confl_pairs, desc=f'[{ds_target}] iterating confl pairs'):
        sets = {ds for ds_pair in confl_pairs[p] for ds in ds_pair if ds in rel_datasets}
        target_ds_setup = setup_dict_rev[ds_target]
        train_datasets = df.loc[df['split_type'] == 'train', 'dataset_id'].unique().tolist()
        target_ds_setup_in_train = any(setup_dict_rev[ds] == target_ds_setup for ds in train_datasets if ds in setup_dict_rev)
        setup_orders = {}
        s1, s2 = sorted(p)
        for ds in sets:
            order = get_pair_order(s1, s2, ds, rts, void_info[ds], epsilon=epsilon)
            if order is None:
                continue
            if ds not in setup_dict_rev:
                no_setup.append(ds)
                continue
            else:
                setup = setup_dict_rev[ds]
            setup_orders.setdefault(setup, set()).add(order)
        if (len({order for orders in setup_orders.values() for order in orders}) < 2):
            # no actual conflicts with the datasets provided
            records.append(dict(s1=s1, s2=s2, mean_order=list({order for orders in setup_orders.values() for order in orders}),
                                informative_here=False,
                                has_characterstic_setups=False, num_characteristic_setups=0, has_unique_setups=False,
                                num_unique_setups=0, has_contradicting_setups=False, num_contradicting_setups=0,
                                target_ds_setup_in_train=target_ds_setup_in_train, target_ds_unique=False,
                                target_ds_characteristic=False, target_ds_contradictory=False,
                                target_ds_contradictory_strict=False,
                                ))
            continue
        accountable_orders = [list(v)[0] for v in setup_orders.values() if len(v) == 1]
        mean_order = np.mean(accountable_orders) if len(accountable_orders) > 0 else None
        characteristic_setups = {}
        if mean_order is not None and mean_order != 0.5:
            for setup, orders in setup_orders.items():
                if len(orders) == 1 and (order:=list(orders)[0]) != np.round(mean_order, 0):
                    characteristic_setups[setup] = order
        # unique?
        unique_setups = {}
        if all([len(v) == 1 for v in setup_orders.values()]):
            rev_lookup = {}
            for setup, orders in setup_orders.items():
                assert len(orders) == 1
                order = list(orders)[0]
                rev_lookup.setdefault(order, set()).add(setup)
            for order in rev_lookup:
                if len(rev_lookup[order]) == 1:
                    unique_setups[list(rev_lookup[order])[0]] = order
        contradictory_setups = {k for k, v in setup_orders.items() if len(v) > 1}
        # check each dataset of each contrad. setup to see whether the pair is contrad. even for *exactly the same parameters*
        contradictory_setups_strict_datasets = set()
        for s in contradictory_setups:
            sets = setup_dict[s]
            for ds1, ds2 in combinations(sets, 2):
                if ds2 in setups_strict_lookup[ds1]:
                    if get_pair_order(s1, s2, ds1, rts, void_info[ds1], epsilon=epsilon) != get_pair_order(s1, s2, ds2, rts, void_info[ds2], epsilon=epsilon):
                        contradictory_setups_strict_datasets.add(ds1)
                        contradictory_setups_strict_datasets.add(ds2)
        target_ds_unique = target_ds_setup in unique_setups
        target_ds_characteristic = target_ds_setup in characteristic_setups
        target_ds_contradictory = target_ds_setup in contradictory_setups
        target_ds_contradictory_strict = ds_target in contradictory_setups_strict_datasets
        records.append(dict(s1=s1, s2=s2, mean_order=mean_order, informative_here=True,
                            characteristic_setups=', '.join(f'{k}:{v}' for k, v in characteristic_setups.items()) if len(characteristic_setups) > 0 else None,
                            unique_setups=', '.join(f'{k}:{v}' for k, v in unique_setups.items()) if len(unique_setups) > 0 else None,
                            contradictory_setups=', '.join(f'{k}' for k in contradictory_setups) if len(contradictory_setups) > 0 else None,
                            has_characterstic_setups=len(characteristic_setups) > 0,
                            num_characteristic_setups=len(characteristic_setups),
                            has_unique_setups=len(unique_setups) > 0,
                            num_unique_setups=len(unique_setups),
                            has_contradicting_setups=len(contradictory_setups) > 0,
                            num_contradicting_setups=len(contradictory_setups),
                            target_ds_setup_in_train=target_ds_setup_in_train,
                            target_ds_unique=target_ds_unique,
                            target_ds_characteristic=target_ds_characteristic,
                            target_ds_contradictory=target_ds_contradictory,
                            target_ds_contradictory_strict=target_ds_contradictory_strict))
    df = pd.DataFrame.from_records(records)
    # print(len(set(map(str, no_setup))), 'datasets could not be mapped to setups')
    return df

def try_inject_setup_info(d, params):
    to_inject = {}
    for c, get_fun in [('H', d.get_hsm_params), ('kPB', lambda r: d.get_tanaka_params(
            r, how=d.tanaka_match, ignore_spp_particle_size=d.tanaka_ignore_spp_particle_size))]:
        if c in params:
            for ds in d.df.dataset_id.unique():
                df = d.df.loc[d.df.dataset_id == ds]
                if (c not in df.columns.tolist()) or pd.isna(df[c]).any():
                    to_inject.setdefault(ds, {}).update(dict(get_fun(df.iloc[0])))
    for ds in to_inject:
        for k, v in to_inject[ds].items():
            d.df.loc[d.df.dataset_id == ds, k] = v

def confl_eval(ds, preds, test_data, train_data, confl_pairs,
               roi_thr=1e-5, epsilon=0.5, setup_params=['column.name', 'ph'],
               Y_debug=None, dataset_iall=None):
    assert len(test_data.df) == len(preds)
    smiles_lookup = {s: i for i, s in enumerate(test_data.df.smiles.tolist())}
    if Y_debug is not None:
        for s, i in smiles_lookup.items():
            assert Y[i] == d.df.iloc[i].rt
    # only pairs from the test dataset
    rel_confl_pairs = {k for k, v in confl_pairs.items()
                       if any(ds in x for x in v)
                       and all(s in test_data.df.smiles.tolist() for s in k)}
    if any(c not in train_data.df.columns.tolist() for c in setup_params):
        print('WARNING: not all info for the setup was found in the train data ({}), trying to inject now...'.format(
        [c for c in setup_params if c not in train_data.df.columns.tolist()]))
        try_inject_setup_info(train_data, setup_params)
    if any(c not in test_data.df.columns.tolist() for c in setup_params):
        print('WARNING: not all info for the setup was found in the test data ({}), trying to inject now...'.format(
        [c for c in setup_params if c not in test_data.df.columns.tolist()]))
        try_inject_setup_info(test_data, setup_params)
    all_data_df = pd.concat([train_data.df, test_data.df]).drop_duplicates('id')
    if (dataset_iall is not None):
        # if we have more information on the datasets, add it
        all_data_df = pd.merge(all_data_df, dataset_iall[['gradient', 'mobilephase', 'authors']],
                               left_on='dataset_id', right_index=True, how='left')
    ds_target_id = test_data.df.dataset_id.unique().item()
    pair_stats_df = get_pair_stats(all_data_df, ds_target_id,
                                   qualifiers=setup_params, confl_pairs=confl_pairs,
                                   void_info=train_data.void_info | test_data.void_info, epsilon=epsilon)
    if pair_stats_df is None:
        return None
    pair_stats_dict = {frozenset(i): dict(r) for i, r in pair_stats_df.set_index(['s1', 's2']).iterrows()}
    records = []
    for c in rel_confl_pairs:
        s1, s2 = sorted(c)
        i1, i2 = smiles_lookup[s1], smiles_lookup[s2]
        rt1 = test_data.df.iloc[i1].rt
        rt2 = test_data.df.iloc[i2].rt
        correct = rt1 - rt2
        if np.abs(correct) < epsilon:
            continue
        correct_order = correct > 0
        roi1 = preds[i1]
        roi2 = preds[i2]
        pred = roi1 - roi2
        if np.abs(pred) < roi_thr:
            pred_order = None
        else:
            pred_order = (pred > 0)
        pred_correct = pred_order == correct_order
        pair_stats_record = pair_stats_dict[c]
        records.append(dict(smiles1=s1, smiles2=s2, correct=pred_correct, rt1=rt1, rt2=rt2, roi1=roi1, roi2=roi2,
                            roi_diff=np.abs(roi1 - roi2), rt_diff=np.abs(rt1 - rt2),
                            setup_in_train=pair_stats_record['target_ds_setup_in_train'],
                            setup_unique=pair_stats_record['target_ds_unique'],
                            setup_characteristic=pair_stats_record['target_ds_characteristic'],
                            setup_contradictory=pair_stats_record['target_ds_contradictory'],
                            any_setup_contradictory=pair_stats_record['has_contradicting_setups'],
                            any_setup_characteristic=pair_stats_record['has_characterstic_setups'],
                            any_setup_unique=pair_stats_record['has_unique_setups'],
                            predictable_from_train=pair_stats_record['target_ds_setup_in_train'] and not pair_stats_record['target_ds_contradictory'],
                            informative_here=pair_stats_record['informative_here']))
    if len(records) == 0:
        return None
    return pd.DataFrame.from_records(records)

if __name__ == '__main__':
    if '__file__' in globals():
        args = EvalArgs().parse_args()
    else:
        args = EvalArgs().parse_args('--model external_test_eval_run --test_sets 0343 0344 /home/fleming/Documents/Uni/RTpred/evaluation/split_datasets/0054_test.tsv /home/fleming/Documents/Uni/RTpred/evaluation/split_datasets/0002_test.tsv /home/fleming/Documents/Uni/RTpred/evaluation/split_datasets/0003_test.tsv /home/fleming/Documents/Uni/RTpred/evaluation/split_datasets/0010_test.tsv /home/fleming/Documents/Uni/RTpred/evaluation/split_datasets/0018_test.tsv /home/fleming/Documents/Uni/RTpred/evaluation/split_datasets/0055_test.tsv /home/fleming/Documents/Uni/RTpred/evaluation/split_datasets/0019_test.tsv --epsilon 10s --test_stats --confl_pairs /home/fleming/Documents/Uni/RTpred/pairs6.pkl --export_rois --get_more_dataset_info'.split())
        args = EvalArgs().parse_args("--model runs/FE_sys/FE_columnph_disjoint_sys_no_fold1_ep10 --test_sets 0004 0017 0018 0048 0049 0052 0079 0080 0101 0158 0179 0180 0181 0182 0226 --epsilon 10s --test_stats --confl_pairs /home/fleming/Documents/Uni/RTpred/pairs6.pkl --overwrite_system_features 'H' 'S*' 'A' 'B' 'C (pH 2.8)' 'C (pH 7.0)' 'kPB' 'αCH2' 'αT/O' 'αC/P' 'αB/P' 'αB/P.1' 'ph' --repo_root_folder /home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/".split())
        args = EvalArgs().parse_args('--model runs/FE_sys/FE_setup_disjoint_sys_yes_cluster_yes_fold1_ep10 --test_sets 0002 0009 0038 0043 0049 0050 0052 0060 0062 0066 0082 0098 0100 0201 0202 0203 0204 0206 0236 0237 0264 0270 0271 0342 0343 0387 --epsilon 10s --test_stats --confl_pairs /home/fleming/Documents/Uni/RTpred/pairs6.pkl'.split())
        args = EvalArgs().parse_args('--model runs/nores/dmpnn_encpv_no_residual3_ep10 --test_sets 0003 0018 0055 0054 0019 0002 --epsilon 10s --test_stats --confl_pairs /home/fleming/Documents/Uni/RTpred/pairs6.pkl --repo_root_folder /home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/ --preds_from_exported_rois runs/dmpnn_encpv_no_residual3_ep10/dmpnn_encpv_no_residual3_ep10_0003_real.tsv runs/dmpnn_encpv_no_residual3_ep10/dmpnn_encpv_no_residual3_ep10_0018_real.tsv runs/dmpnn_encpv_no_residual3_ep10/dmpnn_encpv_no_residual3_ep10_0055_real.tsv runs/dmpnn_encpv_no_residual3_ep10/dmpnn_encpv_no_residual3_ep10_0054_real.tsv runs/dmpnn_encpv_no_residual3_ep10/dmpnn_encpv_no_residual3_ep10_0019_real.tsv runs/dmpnn_encpv_no_residual3_ep10/dmpnn_encpv_no_residual3_ep10_0002_real.tsv --overwrite_system_features'.split() + ['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)', 'kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1', 'ph'])

    if (args.verbose):
        basicConfig(level=INFO)
    if (args.model_type != 'ranknet' and args.gpu):
        torch.set_default_device('cuda')

    # load model
    info('load model...')
    model, data, config = load_model(args.model, args.model_type)
    features_type = parse_feature_spec(config['args']['feature_type'])['mode']
    features_add = config['args']['add_descs']
    n_thr = config['args']['num_features']
    # change paths if necessary for loading additional data
    if (not os.path.exists(data.repo_root_folder)):
        data.repo_root_folder = args.repo_root_folder
    info('load cache')
    # load cached descriptors
    if (args.cache_file is not None):
        features.write_cache = False # flag for reporting changes to cache
        info('load cache')
        if (os.path.exists(args.cache_file)):
            features.cached = pickle.load(open(args.cache_file, 'rb'))
        else:
            features.cached = {}
            warning('cache file does not exist yet')

    test_stats = []
    data_args = {'use_compound_classes': data.use_compound_classes,
                 'use_system_information': data.use_system_information,
                 'metadata_void_rt': (not args.no_metadata_void_rt),
                 'remove_void_compounds': args.remove_void_compounds,
                 'void_factor': args.void_factor,
                 'classes_l_thr': data.classes_l_thr,
                 'classes_u_thr': data.classes_u_thr,
                 'use_usp_codes': data.use_usp_codes,
                 'custom_features': data.descriptors,
                 'use_hsm': data.use_hsm,
                 'use_ph': data.use_ph,
                 'use_gradient': data.use_gradient,
                 'use_newonehot': data.use_newonehot,
                 'repo_root_folder': args.repo_root_folder,
                 'custom_column_fields': data.custom_column_fields,
                 'columns_remove_na': False,
                 'hsm_fields': data.hsm_fields,
                 'graph_mode': args.model_type != 'ranknet',
                 'encoder': (config['args']['mpn_encoder'] if 'mpn_encoder' in config['args']
                             else 'dmpnn'),
                 'remove_doublets': True}
    if (hasattr(data, 'use_tanaka')):
        data_args['use_tanaka'] = data.use_tanaka
    if (hasattr(data, 'tanaka_fields')):
        data_args['tanaka_fields'] = data.tanaka_fields
    if (hasattr(data, 'sys_scales')):
        data_args['sys_scales'] = data.sys_scales
    if (hasattr(data, 'solvent_order')):
        data_args['solvent_order'] = data.solvent_order
    if (hasattr(data, 'use_column_onehot')):
        data_args['use_column_onehot'] = data.use_column_onehot
    info('model preprocessing done')
    if (args.confl_pairs):
        info('loading conflicting pairs')
        confl_pairs = pickle.load(open(args.confl_pairs, 'rb'))
        print(f'number of conflicting pairs loaded: {len(confl_pairs)}')
        # filter confl pairs to only consider datasets from either the train or test set
        train_sets = data.df.iloc[data.train_indices].dataset_id.unique().tolist()
        relevant_sets = set(train_sets) | set(args.test_sets)
        confl_pairs = {k: {ds_pair for ds_pair in v if all(ds in relevant_sets for ds in ds_pair)}
                       for k, v in confl_pairs.items()}
        confl_pairs = {k: v for k, v in confl_pairs.items() if len(v) > 0}
        print(f'only keeping those that conflict for any dataset from train/test data, leaving: {len(confl_pairs)}')
    else:
        confl_pairs = None
    if (args.get_more_dataset_info):
        import sys
        sys.path.append(args.repo_root_folder)
        from pandas_dfs import get_dataset_df
        dataset_iall = get_dataset_df()
    else:
        dataset_iall = None
    for ds in args.test_sets:
        info(f'loading data for {ds}')
        d = Data(**data_args)
        if (not re.match(r'\d{4}', ds)):
            # raw file
            d.add_external_data(ds, void_rt=args.void_rt,
                                       isomeric=(not args.no_isomeric),
                                       split_type='evaluate')
            # for specific eval scenarios it can make sense to use external files for RepoRT datasets.
            # in these cases, the RepoRT ID (if in the filename) can still be used for confl. stats
            # if ((match:=re.search(r'\b(\d{4})_', ds)) is not None):
            #     ds_report_id = match.groups()[0]
            # else:
            #     ds_report_id = None
        else:
            d.add_dataset_id(ds,
                             repo_root_folder=args.repo_root_folder,
                             void_rt=args.void_rt,
                             isomeric=(not args.no_isomeric),
                             split_type='evaluate')
            # ds_report_id = ds
        if (args.remove_train_compounds):
            info('removing train compounds')
            train_compounds_all = set(data.df[args.compound_identifier])
            this_column = d.df['column.name'].values[0]
            train_compounds_col = set(data.df.loc[data.df['column.name'] == this_column, args.compound_identifier])
            if (args.remove_train_compounds_mode == 'print'):
                print('compounds overlap to training data: '
                      + f'{len(set(d.df[args.compound_identifier]) & train_compounds_all) / len(set(d.df[args.compound_identifier])) * 100:.0f}% (all), '
                      + f'{len(set(d.df[args.compound_identifier]) & train_compounds_col) / len(set(d.df[args.compound_identifier])) * 100:.0f}% (same column)')
            else:
                if (args.remove_train_compounds_mode == 'all'):
                    train_compounds = train_compounds_all
                elif (args.remove_train_compounds_mode == 'column'):
                    train_compounds = train_compounds_col
                else:
                    raise NotImplementedError(args.remove_train_compounds_mode)
                prev_len = len(d.df)
                d.df = d.df.loc[~d.df[args.compound_identifier].isin(train_compounds)]
                if args.verbose:
                    print(f'{ds} evaluation: removed {prev_len - len(d.df)} compounds also appearing '
                          f'in the training data (now {len(d.df)} compounds)')
        if (len(d.df) < 2):
            print(f'too few compounds ({len(d.df)}), skipping ...')
            continue
        info('computing features')
        d.compute_features(verbose=args.verbose, mode=features_type, add_descs=features_add,
                           add_desc_file=args.add_desc_file, n_thr=n_thr)
        if (args.model_type != 'ranknet'):
            info('computing graphs')
            d.compute_graphs()
        info('(fake) splitting data')
        d.split_data((0, 0))
        if (hasattr(data, 'descriptor_scaler') or hasattr(data, 'sysfeature_scaler')):
            info('standardize data')
            # perhaps only feature scale and no sysf scaler or vice-versa
            desc_scaler = data.descriptor_scaler if hasattr(data, 'descriptor_scaler') else None
            sys_scaler = data.sysfeature_scaler if hasattr(data, 'sysfeature_scaler') else None
            d.standardize(other_descriptor_scaler=desc_scaler, other_sysfeature_scaler=sys_scaler,
                          can_create_new_scaler=False)
        ((train_graphs, train_x, train_sys, train_y),
         (val_graphs, val_x, val_sys, val_y),
         (test_graphs, test_x, test_sys, test_y)) = d.get_split_data()
        X = np.concatenate((train_x, test_x, val_x)).astype(np.float32)
        X_sys = np.concatenate((train_sys, test_sys, val_sys)).astype(np.float32)
        Y = np.concatenate((train_y, test_y, val_y))
        if (args.confl_pairs is not None):
            rel_confl_pairs = {k for k, v in confl_pairs.items()
                               if any(ds in x for x in v)
                               and all(s in d.df.smiles.tolist() for s in k)}
            rel_confl = {_ for x in rel_confl_pairs for _ in x}
            confl = [smiles in rel_confl for smiles in d.df.smiles]
        info(f'done preprocessing. predicting...')
        if (len(args.preds_from_exported_rois) > 0):
            rois_df = pd.read_csv(args.preds_from_exported_rois[args.test_sets.index(ds)], sep='\t',
                                  names=['smiles', 'rt', 'roi'], header=None)
            d.df = pd.merge(d.df, rois_df, on='smiles', how='left', suffixes=('', '_from_roi'))
            assert (d.df.dropna(subset=['rt'])['rt'] == d.df.dropna(subset=['rt'])['rt_from_roi']).all()
            preds = d.df.roi
        elif (args.model_type == 'mpn' or args.model_type == 'rankformer'):
            # NOTE: for rankformer only works with the `transformer_individual_cls` setting
            graphs = np.concatenate((train_graphs, test_graphs, val_graphs))
            add_sys_features = hasattr(model, 'add_sys_features') and model.add_sys_features
            include_special_features = (hasattr(model, 'include_special_atom_features')
                                        and model.include_special_atom_features)
            if (add_sys_features or include_special_features):
                from utils_newbg import sysfeature_graph
                from utils_newbg import SPECIAL_FEATURES_SIZE
                if add_sys_features:
                    info('add system features to graphs')
                if include_special_features:
                    info('add special atom features to graphs')
                smiles_list = d.df.iloc[np.concatenate((d.train_indices, d.test_indices, d.val_indices))]['smiles'].tolist()
                assert len(graphs) == len(smiles_list)
                assert np.isclose(Y, d.df.iloc[np.concatenate((d.train_indices, d.test_indices, d.val_indices))]['rt'].tolist()).all()
                from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
                extra_dim = (train_sys.shape[1] if add_sys_features else 0) + (SPECIAL_FEATURES_SIZE if include_special_features else 0)
                if (model.add_sys_features_mode == 'bond'):
                    set_extra_bond_fdim(extra_dim)
                elif (model.add_sys_features_mode == 'atom'):
                    set_extra_atom_fdim(extra_dim)
                for i in range(len(graphs)):
                    graphs[i] = sysfeature_graph(smiles_list[i], graphs[i],
                                                 X_sys[i] if add_sys_features else None,
                                                 bond_or_atom=model.add_sys_features_mode,
                                                 special_features=include_special_features)
            if (not args.model_type == 'rankformer' and args.export_embeddings):
                preds, embeddings = model.predict(graphs, X, X_sys, batch_size=args.batch_size,
                                                  prog_bar=args.verbose, ret_features=True)
                embeddings_df = pd.DataFrame({'smiles': d.df.smiles} |
                                             {f'e{i}': embeddings[:, i]
                                              for i in range(embeddings.shape[1])})
                ds_id = os.path.basename(ds) if not re.match(r'\d{4}', ds) else ds
                embeddings_df.to_csv(f'runs/{config["name"]}_{ds_id}_embeddings.tsv',
                                     sep='\t')
            else:
                preds = model.predict(graphs, X, X_sys, batch_size=args.batch_size,
                                      **(dict(ret_features=False, prog_bar=args.verbose) if not args.model_type == 'rankformer' else {}))
        elif (args.model_type == 'rankformer_rt'):
            from ranknet_transformer import rankformer_rt_predict
            graphs = np.concatenate((train_graphs, test_graphs, val_graphs))
            preds = rankformer_rt_predict(model, graphs, X, X_sys, batch_size=args.batch_size,
                                          prog_bar=args.verbose)
        else:
            preds = predict(X, model, args.batch_size)
        info('done predicting. evaluation...')
        print(ds, d.void_info[ds])
        acc = eval_(Y, preds, args.epsilon, void_rt=d.void_info[ds], dont_count_low_epsilon=False)
        acc_ignore_epsilon = eval_(Y, preds, args.epsilon, void_rt=d.void_info[ds], dont_count_low_epsilon=True)
        optional_stats = {}
        if (confl_pairs is not None and not args.no_optional_stats):
            optional_stats['acc_confl'] = eval_(Y[confl], preds[confl], args.epsilon, void_rt=d.void_info[ds],
                                                dont_count_low_epsilon=False) if any(confl) else np.nan
            optional_stats['acc_ignore_epsilon_confl'] = eval_(Y[confl], preds[confl], args.epsilon, void_rt=d.void_info[ds],
                                                               dont_count_low_epsilon=True) if any(confl) else np.nan
            optional_stats['acc_nonconfl'] = eval_(Y[~np.array(confl)], preds[~np.array(confl)], args.epsilon, void_rt=d.void_info[ds],
                                                   dont_count_low_epsilon=False) if any(confl) else acc
            optional_stats['acc_ignore_epsilon_nonconfl'] = eval_(Y[~np.array(confl)], preds[~np.array(confl)], args.epsilon, void_rt=d.void_info[ds],
                                                                  dont_count_low_epsilon=True) if any(confl) else acc
            optional_stats['num_confl'] = np.array(confl).sum()
            if (args.overwrite_system_features is not None and len(args.overwrite_system_features) > 0):
                system_features = args.overwrite_system_features
            else:
                system_features = data.system_features
            print('making detailed stats for system features:', ', '.join(system_features))
            confl_stats_df = confl_eval(ds, preds=preds, test_data=d, train_data=data, confl_pairs=confl_pairs,
                                        epsilon=args.epsilon, setup_params=system_features, Y_debug=Y, dataset_iall=dataset_iall)
            if (len(system_features) > 0 and confl_stats_df is not None):
                optional_stats['acc_confl_predictable_from_train'] = (confl_stats_df.loc[confl_stats_df.predictable_from_train, 'correct']).mean()
                optional_stats['acc_confl_really_informative'] = (confl_stats_df.loc[confl_stats_df.informative_here, 'correct']).mean()
                optional_stats['num_confl_setup_unique'] = confl_stats_df.setup_unique.sum()
                optional_stats['num_confl_setup_characteristic'] = confl_stats_df.setup_characteristic.sum()
                optional_stats['acc_confl_setup_unique'] = (confl_stats_df.loc[confl_stats_df.setup_unique, 'correct']).mean()
                optional_stats['acc_confl_setup_characteristic'] = (confl_stats_df.loc[confl_stats_df.setup_characteristic, 'correct']).mean()
                optional_stats['acc_confl_noncontradictory'] = (confl_stats_df.loc[~confl_stats_df.any_setup_contradictory, 'correct']).mean()
                # TODO: more stats?
        d.df['roi'] = preds[np.arange(len(d.df.rt))[ # restore correct order
            np.argsort(np.concatenate([d.train_indices, d.test_indices, d.val_indices]))]]
        if (not args.include_void_compounds_mcd):
            df_mcd = d.df.loc[d.df.rt > d.void_info[ds]]
        else:
            df_mcd = d.df
        mcd = len(df_mcd) - lcs_results(df_mcd, args.mcd_method)
        mcd_ratio = mcd / (len(df_mcd) - 1) # subtract one because a single compound cannot be in conflict

        # acc2, results = eval2(d.df, args.epsilon)
        if (args.classyfire):
            info('computing classyfire stats')
            classyfire_stats(d, args, compound_identifier=args.compound_identifier)
        if (args.dataset_stats): # NOTE: DEPRECATED?
            raise Exception('Deprecated for now')
            info('computing dataset stats')
            dataset_stats(d)
            compound_stats(d, args)
            pair_stats(d, True)
            pass
        if (args.test_stats):
            info('computing test stats')
            stats = data_stats(d, data, data.custom_column_fields, compound_identifier=args.compound_identifier)
            stats.update({'acc': acc, 'id': ds, 'mcd': mcd, 'mcd_ratio': mcd_ratio, 'acc_ignore_epsilon': acc_ignore_epsilon,
                          'lcs_dist': mcd # legacy
                          })
            stats.update(optional_stats)
            test_stats.append(stats)
        else:
            print(f'{ds}: {acc:.3f}, MCD {mcd:.0f}, MCD ratio {mcd_ratio:.3f}, acc_ignore_epsilon {acc_ignore_epsilon:.3f} \t (#data: {len(Y)})')
        if (args.diffs):
            info('computing outlier stats')
            df = rt_roi_diffs(d, Y, preds)
            # with pd.option_context('display.max_rows', None):
            #     print(df.sort_values(by='rt')[['id', 'rt', 'roi', 'diffs']])
            print('outliers:')
            print(df.loc[df.diffs == 1, ['id', 'roi', 'rt', 'rt_gam']])
            if (args.plot_diffs):
                visualize_df(df)
        if (args.export_rois):
            info('exporting ROIs')
            # TODO: don't overwrite
            if (not re.match(r'\d{4}', ds)):
                ds = os.path.basename(ds)
            model_spec = os.path.basename(args.model) # preserve epoch specification if present
            if (args.export_rois_dir is None):
                roi_dir = f'runs/{model_spec}'
            else:
                roi_dir = args.export_rois_dir
            export_predictions(d, preds, f'{roi_dir}/{model_spec}_{ds}.tsv')
        if (False and args.classyfire):
            fig = px.treemap(d.df.dropna(subset=['classyfire.kingdom', 'classyfire.superclass', 'classyfire.class']),
                             path=['classyfire.kingdom', 'classyfire.superclass', 'classyfire.class'],
                             title=f'{ds} data ({acc:.2%} accuracy)')
            fig.show(renderer='browser')
    if (args.test_stats and len(test_stats) > 0):
        test_stats_df = pd.DataFrame.from_records(test_stats, index='id')
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 500)
        print(test_stats_df)
        print(test_stats_df[sorted([c for c in test_stats_df.columns if c.startswith('acc')])].agg(['mean', 'median']))
        if (args.output is not None):
            json.dump({t['id']: t for t in test_stats},
                      open(args.output, 'w'), indent=2, cls=NpEncoder)
