import logging
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from rdkit import rdBase
import pickle
import json
import os
import re
import contextlib
from tap import Tap
from typing import List, Literal, Optional, Union
import pandas as pd
from collections import defaultdict
import sys

from utils import Data
from features import features, parse_feature_spec
from evaluate import predict, export_predictions, load_model
from utils_newbg import RankDataset, check_integrity
from sampling import CustomWeightedRandomSampler, calc_sampling_weights

logger = logging.getLogger('2-step')
info = logger.info

def time_to_min(timestr):
    timestr = str(timestr)
    if (match_ := re.match(r'[\d\.]+ *(min|s)', timestr)):
        unit = match_.groups()[0]
        if unit == 's':
            timestr = float(timestr.replace('s', '').strip()) / 60
        elif unit == 'min':
            timestr = float(timestr.replace('min', '').strip())
        else:
            raise ValueError(f'wrong unit for epsilon ({timestr}): {unit}')
    elif (re.match(r'[\d\.]+', timestr)):
        timestr = float(timestr.strip())
    else:
        raise ValueError(f'wrong format for epsilon ({timestr})')
    return timestr

def none_arg(none_arg):
    if (none_arg is None or str(none_arg).lower() == 'none'):
        none_arg = None
    else:
        try:
            none_arg = int(none_arg)
        except:
            raise ValueError(f'{none_arg=}')
    return none_arg


class TrainArgs(Tap):
    input: List[str]            # Either CSV or dataset ids
    model_type: Literal['mpn'] = 'mpn'
    feature_type: Literal['None', 'rdkall', 'rdk2d', 'rdk3d'] = 'None' # type of features to use
    # training
    gpu: bool = False
    batch_size: int = 512
    epochs: int = 10
    early_stopping_patience: Optional[int] = None # stop training when val loss doesn't improve for this number of times
    test_split: float = 0                         # not needed when testing on exclusive test datasets afterwards
    val_split: float = 0.05
    device: Optional[str] = None  # either `mirrored` or specific device name like gpu:1 or None (auto)
    remove_test_compounds: List[str] = [] # remove compounds occurring in the specified (test) datasets
    remove_test_compounds_mode: Literal['exact', '2d'] = '2d' # remove exact structures or those with same canonical SMILES
    remove_test_compounds_rarest: bool = False # only remove rarest 50 percent of test compounds
    exclude_compounds_list: Optional[str] = None # list of compounds to exclude from training
    learning_rate: float = 5e-4
    adaptive_learning_rate: bool = False
    no_encoder_train: bool = False # don't train the encoder(embedding) layers
    # data
    no_isomeric: bool = False # do not use isomeric data (if available)
    balance: bool = False       # balance data by dataset
    no_group_weights: bool = False # don't scale weights by number of dataset pairs; use this option when *sampling*
    cluster: bool = False          # cluster datasets with same column params for calculating group weights
    downsample_groups: bool = False       # min number of pairs will be used as the max pair nr for each group
    downsample_always_confl: bool = False # include all conflicting pairs also when downsampling
    downsample_factor: float=1.0          # if greater than 1, some clusters may have less pairs
    group_weights_only_intra_cluster: bool=False  # group-weights are used, but only for weighing within a cluster
    sample: bool=False                            # sample the RankDataset based on group weights
    sampling_count: int=500_000                      # how many pairs per epoch when using the `sample` option
    sampling_mode: Literal['compounds', 'pairs']='pairs' # compute sampling probabilities based on dataset compounds or pairs
    sampling_sqrt_weights: bool=False                    # use sqrt on compounds/pair counts to prevent extreme probability distributions
    void_rt: float = 0.0        # void time threshold; used for ALL datasets (if > 0)
    no_metadata_void_rt: bool = False # do not use t0 value from repo metadata (times void_factor)
    remove_void_compounds: bool = False # throw out all compounds eluting in the void volume
    void_factor: float = 2              # factor for 'column.t0' value to use as void threshold
    void_extra_file: Optional[str] = None # extra tsv file with dataset ID as first column and void rt guess as second; no header
    validation_datasets: List[str] = [] # datasets to use for validation (instead of split of training data)
    test_datasets: List[str] = [] # datasets to use for test (instead of split of training data)
    # features
    features: List[str] = []                                     # custom descriptors
    no_standardize: bool = False                                    # do not standardize system features + descriptors
    reduce_features: bool = False                                    # reduce features
    num_features: Optional[int] = None
    # additional features
    sysinfo: bool = False       # use column information as add. features
    columns_use_hsm: bool = False
    columns_use_tanaka: bool = False
    columns_use_onehot: bool = False
    hsm_fields: List[str] = ['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)']
    tanaka_fields: List[str] = ['kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1']
    custom_column_fields: List[str] = []
    fallback_column: str = 'Waters ACQUITY UPLC BEH C18' # column data to use when needed and no data available; can also be 'average'
    fallback_metadata: str = '0045' # repository metadata to use when needed and no data available; can also be 'average' or 'zeros'
    usp_codes: bool = False     # use column usp codes as onehot system features (only for `--sysinfo`)
    use_ph: bool = False        # use pH estimations of mobilephase if available
    use_gradient: bool = False  # use mobile phase solvent concentrations at specific gradient positions WARNING: can lead to rt being leaked
    debug_onehot_sys: bool = False # onehot dataset encoding
    onehot_test_sets: List[str] = [] # test set IDs to include in onehot encoding
    columns_use_newonehot: bool = False
    tanaka_match: Literal['best_match', 'exact'] = 'best_match' # 'exact': only allow tanaka parameters with the matching particle size
    tanaka_ignore_spp_particle_size: bool = True
    # model general
    sizes: List[int] = [256, 65] # hidden layer sizes for ranking: [mol, sysxmol] -> ROI
    sizes_sys: List[int] = [256, 256] # hidden layer sizes for system feature vs. molecule encoding
    encoder_size: int = 512 # MPNencoder size
    mpn_depth: int = 3      # Number of message-passing steps
    dropout_rate_encoder: float = 0.0   # MPN dropout rate
    dropout_rate_pv: float = 0.0   # system preference encoding dropout rate
    dropout_rate_rank: float = 0.0   # final ranking layers dropout rate
    # mpn model
    mpn_loss: Literal['margin', 'bce'] = 'margin'
    mpn_margin: float = 0.1
    mpn_encoder: Literal['dmpnn'] = 'dmpnn'
    smiles_for_graphs: bool = False # always use SMILES internally, compute graphs only on demand
    mpn_no_residual_connections_encoder: bool = False # last stack for mpn model only takes the encoding convolved with sys features
    mpn_add_sys_features: bool = False                # add sys features to the graphs themselves
    mpn_add_sys_features_mode: Literal['bond', 'atom'] = 'atom' # whether to add sys featues as 'bond' and 'atom' features
    mpn_no_sys_layers: bool = False # don't add any layers for sys features to the MPN (for example when sys features are already part of the graphs)
    mpn_sys_blowup: bool = False # extra layer which blows up sysfeatures dimension to encoder size
    mpn_no_sigmoid_roi: bool = False # don't use sigmoid as last step for keeping ROI in range [0, 1]
    # pairs
    epsilon: Union[str, float] = '10s' # difference in evaluation measure below which to ignore falsely predicted pairs
    pair_step: int = 1
    pair_stop: Optional[Union[int, str]] = None
    no_rtdiff_pair_weights: bool=False            # don't weigh pairs according to rt difference
    weight_steep: float = 20
    weight_mid: float = 0.75
    dynamic_weights: bool = False # adapt epsilon/weights to gradient length
    discard_smaller_than_epsilon: bool = False # don't weigh by rt diff; simply discard any pairs with rt_diff<epsilon
    inter_pairs: bool = False # use pairs of compounds of different datasets (DEPRECATED)
    no_intra_pairs: bool = False # don't use pairs of compounds of the same dataset
    max_pair_compounds: Optional[int] = None
    max_num_pairs: Optional[int] = None            # limit for the number of pairs per dataset/group
    conflicting_smiles_pairs: Optional[str] = None # pickle file with conflicting pairs (smiles)
    confl_weight: float = 1.                       # weight modifier for conflicting pairs
    check_data: bool = False                       # check how many pairs are conflicting/unpredictable
    clean_data: bool = False                       # remove unpredictable pairs
    # data locations
    repo_root_folder: str = '../RepoRT/' # location of RepoRT
    add_desc_file: str = 'data/qm_merged.csv'
    cache_file: str = 'cached_descs.pkl'
    # output control
    verbose: bool = False
    no_progbar: bool = False
    run_name: Optional[str] = None
    export_rois: bool = False
    save_data: bool = False
    ep_save: bool = False       # save after each epoch
    no_train_acc_all: bool = False # can save memory; this metric is pretty useless anyways
    no_train_acc: bool = False # can save memory; this metric is pretty useless anyways

    def configure(self) -> None:
        self.add_argument('--epsilon', type=time_to_min)
        self.add_argument('--pair_stop', type=none_arg)

def generic_run_name():
    from datetime import datetime
    time_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    return f'2-step_{time_str}'


def preprocess(data: Data, args: TrainArgs):
    data.compute_features(**parse_feature_spec(args.feature_type), n_thr=args.num_features, verbose=args.verbose)
    if (data.train_y is not None):
        # assume everything was computed, split etc. already
        return ((data.train_graphs, data.train_x, data.train_sys, data.train_y),
                (data.val_graphs, data.val_x, data.val_sys, data.val_y),
                (data.test_graphs, data.test_x, data.test_sys, data.test_y))
    if (args.cache_file is not None and hasattr(features, 'write_cache')
        and features.write_cache):
        info('writing cache, don\'t interrupt!!')
        pickle.dump(features.cached, open(args.cache_file, 'wb'))
    if args.debug_onehot_sys:
        sorted_dataset_ids = sorted(set(args.input) | set(args.onehot_test_sets))
        data.compute_system_information(True, sorted_dataset_ids)
    info('done. preprocessing...')
    if (data.graph_mode):
        data.compute_graphs()
    data.split_data((args.test_split, args.val_split))
    if (not args.no_standardize):
        data.standardize()
    if (args.reduce_features):
        data.reduce_f()
    if (args.fallback_metadata == 'average' or args.fallback_column == 'average'):
        data.nan_columns_to_average()
    if (args.fallback_metadata == 'zeros' or args.fallback_column == 'zeros'):
        data.nan_columns_to_zeros()
    return data.get_split_data((args.test_split, args.val_split))

def rename_old_writer_logs(prefix):
    suffixes = ['_train', '_val', '_confl']
    if (any(os.path.exists(prefix + suffix) for suffix in suffixes)):
        from datetime import datetime
        stamp = datetime.fromtimestamp(os.path.getmtime(
            [prefix + suffix for suffix in suffixes if os.path.exists(prefix + suffix)][0]
        )).strftime('%Y%m%d_%H-%M-%S')
        for suffix in suffixes:
            if os.path.exists(prefix + suffix):
                new_dir = prefix + '_' + stamp + suffix
                os.rename(prefix + suffix, new_dir)
                print(f'old logdir {prefix + suffix} -> {new_dir}')

if __name__ == '__main__':
    # arguments can be read directly from JSON
    if (len(sys.argv) == 2 and (json_file:=sys.argv[1]).endswith('.json')):
        args = TrainArgs().from_dict(json.load(open(json_file))['args'])
    else:
        args = TrainArgs().parse_args()
    if (args.run_name is None):
        run_name = generic_run_name()
        print(f'preparing ROI prediction model "{run_name}"')
    else:
        run_name = args.run_name
    # logging
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    if (args.verbose):
        logger.setLevel(logging.INFO)
        logging.getLogger('2-step.utils').setLevel(logging.INFO)
        fh = logging.FileHandler(run_name + '.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s'))
        logger.addHandler(fh)
        ch.setLevel(logging.INFO)
    else:
        rdBase.DisableLog('rdApp.warning')
    # importing training libraries, setting associated parameters
    if (args.model_type == 'mpn'):
        from mpnranker2 import MPNranker, train as mpn_train
        import torch
        if (args.gpu):
            torch.set_default_device('cuda')
        print('torch device:', torch.tensor([1.2, 3.4]).device, file=sys.stderr)
        graphs = True
    else:
        raise NotImplementedError(args.model_type)
    # additional parameters taken from args
    y_neg = (args.mpn_loss == 'margin')
    # caching
    if (args.cache_file is not None and args.feature_type != 'None'):
        features.write_cache = False # flag for reporting changes to cache
        info('reading in cache...')
        if (os.path.exists(args.cache_file)):
            features.cached = pickle.load(open(args.cache_file, 'rb'))
        else:
            features.cached = {}
            info('cache file does not exist yet')
    info('reading in data and computing features...')
    # additional data from special files
    void_guesses = {}
    if (args.void_extra_file is not None):
        for line in open(args.void_extra_file).readlines():
            ds, void_guess = line.strip().split('\t')
            void_guesses[ds] = float(void_guess)
    # TRAINING
    if (len(args.input) == 1 and os.path.exists(input_ := args.input[0]) and re.match(r'.*\.(tf|pt)$', input_)):
        if (input_.endswith('.tf')):
            print('input is trained Tensorflow model')
            raise NotImplementedError('Tensorflow model')
        elif (input_.endswith('.pt')):
            print('input is trained PyTorch model')
            ranker, data, config = load_model(input_, 'mpn')
    else:
        print('input from RepoRT dataset IDs and/or external datasets')
        data = Data(use_system_information=args.sysinfo,
                    metadata_void_rt=(not args.no_metadata_void_rt),
                    remove_void_compounds=args.remove_void_compounds,
                    void_factor=args.void_factor,
                    use_usp_codes=args.usp_codes, custom_features=args.features,
                    use_hsm=args.columns_use_hsm, use_tanaka=args.columns_use_tanaka,
                    use_newonehot=args.columns_use_newonehot, use_ph=args.use_ph,
                    use_column_onehot=args.columns_use_onehot,
                    use_gradient=args.use_gradient,
                    repo_root_folder=args.repo_root_folder,
                    custom_column_fields=args.custom_column_fields,
                    hsm_fields=args.hsm_fields, tanaka_fields=args.tanaka_fields,
                    tanaka_match=args.tanaka_match,
                    tanaka_ignore_spp_particle_size=args.tanaka_ignore_spp_particle_size,
                    graph_mode=graphs, smiles_for_graphs=args.smiles_for_graphs,
                    fallback_column=args.fallback_column,
                    fallback_metadata=args.fallback_metadata,
                    encoder=args.mpn_encoder)
        for did, split_type in (list(zip(args.input, ['train'] * len(args.input)))
                                + list(zip(args.validation_datasets, ['val'] * len(args.validation_datasets)))
                                + list(zip(args.test_datasets, ['test'] * len(args.test_datasets)))):
            if re.match(r'\d{4}', did):
                # RepoRT dataset
                data.add_dataset_id(did,
                                    repo_root_folder=args.repo_root_folder,
                                    void_rt=void_guesses.get(did, args.void_rt),
                                    isomeric=(not args.no_isomeric),
                                    split_type=split_type)
            elif os.path.exists(did):
                # external dataset
                data.add_external_data(did, metadata_void_rt=(not args.no_metadata_void_rt), void_rt=void_guesses.get(did, args.void_rt),
                                       isomeric=(not args.no_isomeric), split_type=split_type)
            else:
                raise Exception(f'input {did} not supported')
        if (args.remove_test_compounds is not None and len(args.remove_test_compounds) > 0):
            d_temp = Data()
            for t in args.remove_test_compounds:
                d_temp.add_dataset_id(t, repo_root_folder=args.repo_root_folder,
                                      isomeric=(not args.no_isomeric))
            if (args.remove_test_compounds_mode == '2d'):
                data.df['inchikey1'] = data.df['inchikey.std'].apply(lambda i: i.split('-')[0])
                d_temp.df['inchikey1'] = d_temp.df['inchikey.std'].apply(lambda i: i.split('-')[0])
                compounds_id_remove = 'inchikey1'
            else:
                compounds_id_remove = 'smiles'
            if (args.remove_test_compounds_rarest):
                # compound occurences
                occs = defaultdict(int)
                for c in d_temp.df[compounds_id_remove].unique():
                    occs[c] = data.df.loc[data.df[compounds_id_remove] == c, 'dataset_id'].nunique()
                compounds_to_remove = list(sorted(d_temp.df[compounds_id_remove].tolist(), key=lambda x: occs[x]))[:int(len(d_temp.df) / 2)]
            else:
                compounds_to_remove = set(d_temp.df[compounds_id_remove].tolist())
            len_orig = data.df[compounds_id_remove].nunique()
            data.df = data.df.loc[~data.df[compounds_id_remove].isin(compounds_to_remove)]
            print(f'removed {len(compounds_to_remove)} (actually {len_orig - data.df[compounds_id_remove].nunique()}) compounds occuring '
                  'in test data from training data')
        if (args.exclude_compounds_list is not None):
            # exclude everything from exclusion list/table where all columns match
            # e.g., only smiles; or smiles and dataset_id
            to_exclude = pd.read_csv(args.exclude_compounds_list)
            prev_len = len(data.df)
            data.df = pd.merge(data.df, to_exclude, on=to_exclude.columns.tolist(), how='outer',
                               indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)
            print(f'removed {prev_len - len(data.df)} compounds by column(s) {",".join(to_exclude.columns)} '
                  f'from exclusion list (length {len(to_exclude)})')
        if (args.balance and len(args.input) > 1):
            data.balance()
            info('added data for datasets:\n' +
                 '\n'.join([f'  - {did} ({name})' for did, name in
                            set(data.df[['dataset_id', 'column.name']].itertuples(index=False))]))
    ((train_graphs, train_x, train_sys, train_y),
     (val_graphs, val_x, val_sys, val_y),
     (test_graphs, test_x, test_sys, test_y)) = preprocess(data, args)

    if (args.mpn_encoder == 'dmpnn'):
        from mpnranker2 import custom_collate
        from dmpnn_graph import dmpnn_batch
        if (args.mpn_add_sys_features):
            from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
            if (args.mpn_add_sys_features_mode == 'bond'):
                set_extra_bond_fdim(train_sys.shape[1])
            elif (args.mpn_add_sys_features_mode == 'atom'):
                set_extra_atom_fdim(train_sys.shape[1])
        custom_collate.graph_batch = dmpnn_batch
    else:
        raise NotImplementedError(args.mpn_encoder)
    rename_old_writer_logs(f'runs/{run_name}')
    writer = SummaryWriter(f'runs/{run_name}_train')
    val_writer = SummaryWriter(f'runs/{run_name}_val') if len(val_y) > 0 else None
    confl_writer = SummaryWriter(f'runs/{run_name}_confl')
    if (args.save_data):
        pickle.dump(data, open(os.path.join(f'{run_name}_data.pkl'), 'wb'))
        json.dump({'train_sets': args.input, 'name': run_name,
                   'args': args._log_all()},
                  open(f'{run_name}_config.json', 'w'), indent=2)

    conflicting_smiles_pairs = (pickle.load(open(args.conflicting_smiles_pairs, 'rb'))
                                if args.conflicting_smiles_pairs is not None else {})
    info('done. Initializing RankDatasets...')
    print(f'{data.void_info=}')
    print(f'training data shapes: {train_x.shape=}, {train_sys.shape=}')
    traindata = RankDataset(x_mols=train_graphs, x_extra=train_x, x_sys=train_sys,
                            x_ids=data.df.iloc[data.train_indices].smiles.tolist(),
                            y=train_y, x_sys_global_num=data.x_info_global_num,
                            dataset_info=data.df.dataset_id.iloc[data.train_indices].tolist(),
                            void_info=data.void_info,
                            pair_step=args.pair_step,
                            pair_stop=args.pair_stop, use_pair_weights=(not args.no_rtdiff_pair_weights),
                            discard_smaller_than_epsilon=args.discard_smaller_than_epsilon,
                            use_group_weights=(not args.no_group_weights),
                            cluster=args.cluster,
                            downsample_groups=args.downsample_groups,
                            downsample_always_confl=args.downsample_always_confl,
                            downsample_factor=args.downsample_factor,
                            group_weights_only_intra_cluster=args.group_weights_only_intra_cluster,
                            no_inter_pairs=(not args.inter_pairs),
                            no_intra_pairs=args.no_intra_pairs,
                            max_indices_size=args.max_pair_compounds,
                            max_num_pairs=args.max_num_pairs,
                            weight_mid=args.weight_mid,
                            weight_steepness=args.weight_steep,
                            dynamic_weights=args.dynamic_weights,
                            y_neg=y_neg,
                            y_float=('rankformer' in args.model_type),
                            conflicting_smiles_pairs=conflicting_smiles_pairs,
                            confl_weight=args.confl_weight,
                            add_sysfeatures_to_graphs=args.mpn_add_sys_features,
                            sysfeatures_graphs_mode=args.mpn_add_sys_features_mode)
    valdata = RankDataset(x_mols=val_graphs, x_extra=val_x, x_sys=val_sys,
                          x_ids=data.df.iloc[data.val_indices].smiles.tolist(),
                          y=val_y, x_sys_global_num=data.x_info_global_num,
                          dataset_info=data.df.dataset_id.iloc[data.val_indices].tolist(),
                          void_info=data.void_info,
                          pair_step=args.pair_step,
                          pair_stop=args.pair_stop, use_pair_weights=(not args.no_rtdiff_pair_weights),
                          discard_smaller_than_epsilon=args.discard_smaller_than_epsilon,
                          use_group_weights=(not args.no_group_weights),
                          cluster=args.cluster,
                          downsample_groups=args.downsample_groups,
                          downsample_always_confl=args.downsample_always_confl,
                          downsample_factor=args.downsample_factor,
                          group_weights_only_intra_cluster=args.group_weights_only_intra_cluster,
                          no_inter_pairs=(not args.inter_pairs),
                          no_intra_pairs=args.no_intra_pairs,
                          max_indices_size=args.max_pair_compounds,
                          max_num_pairs=args.max_num_pairs,
                          weight_mid=args.weight_mid,
                          weight_steepness=args.weight_steep,
                          dynamic_weights=args.dynamic_weights,
                          y_neg=y_neg,
                          y_float=('rankformer' in args.model_type),
                          conflicting_smiles_pairs=conflicting_smiles_pairs,
                          confl_weight=args.confl_weight,
                          add_sysfeatures_to_graphs=args.mpn_add_sys_features,
                          sysfeatures_graphs_mode=args.mpn_add_sys_features_mode)
    if (args.clean_data or args.check_data):
        print('training data check:')
        stats_train, clean_train, _ = check_integrity(traindata, clean=args.clean_data)
        if (args.clean_data):
            traindata.remove_indices(clean_train)
            print(f'cleaning up {len(clean_train)} of {len(traindata.y_trans)} total '
                  f'({len(clean_train)/len(traindata.y_trans):.0%}) pairs for being invalid')
        print('validation data check:')
        stats_val, clean_val, _ = check_integrity(valdata, clean=args.clean_data)
        if (args.clean_data):
            valdata.remove_indices(clean_val)
            print(f'cleaning up {len(clean_val)} of {len(valdata.y_trans)} total '
                  f'({np.divide(len(clean_val), len(valdata.y_trans)):.0%}) pairs for being invalid')
    if (args.sample):
        sampling_weights_train = calc_sampling_weights(traindata, method=args.sampling_mode, cluster_informed=args.cluster,
                                                       sqrt_weights=args.sampling_sqrt_weights, verbose=args.verbose)
        sampling_weights_val = calc_sampling_weights(valdata, method=args.sampling_mode, cluster_informed=args.cluster,
                                                     sqrt_weights=args.sampling_sqrt_weights, verbose=args.verbose)
        sampler_train = CustomWeightedRandomSampler(sampling_weights_train, args.sampling_count, replacement=True)
        sampler_val = CustomWeightedRandomSampler(sampling_weights_val, args.sampling_count, replacement=True)
    else:
        sampler_train = sampler_val = None
    trainloader = DataLoader(traindata, args.batch_size, shuffle=(not args.sample), sampler=sampler_train,
                             generator=torch.Generator(device='cuda' if args.gpu else 'cpu'),
                             collate_fn=custom_collate)
    valloader = DataLoader(valdata, args.batch_size, shuffle=(not args.sample), sampler=sampler_val,
                           generator=torch.Generator(device='cuda' if args.gpu else 'cpu'),
                           collate_fn=custom_collate) if len(valdata) > 0 else None
    if ('ranker' not in vars() or ranker is None):    # otherwise loaded already
        if (args.model_type == 'mpn'):
            ranker = MPNranker(encoder=args.mpn_encoder,
                               extra_features_dim=train_x.shape[1],
                               sys_features_dim=train_sys.shape[1],
                               hidden_units=args.sizes, hidden_units_pv=args.sizes_sys,
                               encoder_size=args.encoder_size,
                               depth=args.mpn_depth,
                               dropout_rate_encoder=args.dropout_rate_encoder,
                               dropout_rate_pv=args.dropout_rate_pv,
                               dropout_rate_rank=args.dropout_rate_rank,
                               res_conn_enc=(not args.mpn_no_residual_connections_encoder),
                               add_sys_features=args.mpn_add_sys_features,
                               add_sys_features_mode=args.mpn_add_sys_features_mode,
                               no_sys_layers=args.mpn_no_sys_layers,
                               sys_blowup=args.mpn_sys_blowup,
                               no_sigmoid_roi=args.mpn_no_sigmoid_roi)
        else:
            raise NotImplementedError(args.model_type)
        print(ranker)
        print('total params', sum(p.numel() for p in ranker.parameters()))
        print('total params (trainable)', sum(p.numel() for p in ranker.parameters() if p.requires_grad))
    try:
        if (args.model_type == 'mpn'):
            mpn_train(ranker=ranker, bg=trainloader, epochs=args.epochs,
                      epochs_start=ranker.max_epoch,
                      writer=writer, val_g=valloader, val_writer=val_writer,
                      confl_writer=confl_writer, # TODO:
                      steps_train_loss=np.ceil(len(trainloader) / 100).astype(int),
                      steps_val_loss=np.ceil(len(trainloader) / 5).astype(int),
                      batch_size=args.batch_size, epsilon=args.epsilon,
                      sigmoid_loss=(args.mpn_loss == 'bce'), margin_loss=args.mpn_margin,
                      early_stopping_patience=args.early_stopping_patience,
                      learning_rate=args.learning_rate,
                      adaptive_lr=args.adaptive_learning_rate,
                      no_encoder_train=args.no_encoder_train, ep_save=args.ep_save,
                      eval_train_all=(not args.no_train_acc_all),
                      accs=(not args.no_train_acc))
        else:
            raise NotImplementedError(args.model_type)
    except KeyboardInterrupt:
        print('caught interrupt; stopping training')
    if (args.save_data):
        torch.save(ranker, run_name + '.pt')
    if hasattr(ranker, 'predict'):
        train_preds = ranker.predict(train_graphs, train_x.astype(np.float32), train_sys.astype(np.float32),
                                     batch_size=args.batch_size * 2,
                                     prog_bar=args.verbose)
        if (len(val_x) > 0):
            val_preds = ranker.predict(val_graphs, val_x.astype(np.float32), val_sys.astype(np.float32), batch_size=args.batch_size * 2)
        if (len(test_x) > 0):
            test_preds = ranker.predict(test_graphs, test_x.astype(np.float32), test_sys.astype(np.float32), batch_size=args.batch_size * 2)
            if (args.export_rois):
                if not os.path.isdir('runs'):
                    os.mkdir('runs')
                export_predictions(data, test_preds, f'runs/{run_name}_test.tsv', 'test')
    if (args.cache_file is not None and hasattr(features, 'write_cache') and features.write_cache):
        print('writing cache, don\'t interrupt!!')
        pickle.dump(features.cached, open(args.cache_file, 'wb'))
