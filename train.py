from logging import INFO
import numpy as np
import tensorflow as tf
from LambdaRankNN import RankNetNN
from tensorflow.python.platform.tf_logging import get_logger, info
from mpnranker import MPNranker, train as mpn_train, predict as mpn_predict
from tensorboardX import SummaryWriter
from rdkit import rdBase
import pickle
import json
import os
import argparse
import re
import contextlib
from itertools import product
from tap import Tap
from typing import List, Literal, Optional

from utils import BatchGenerator, Data
from features import features, parse_feature_spec
from evaluate import eval_, predict, export_predictions


class TrainArgs(Tap):
    input: List[str]            # Either CSV or dataset ids
    model_type: Literal['ranknet', 'mpn'] = 'ranknet'
    feature_type: Literal['None', 'rdkall', 'rdk2d', 'rdk3d'] = 'rdkall' # type of features to use
    # training
    batch_size: int = 256
    epochs: int = 5
    early_stopping_patience: Optional[int] = None # stop training when val loss doesn't improve for this number of times
    test_split: float = 0.2
    val_split: float = 0.05
    device: Optional[str] = None  # either `mirrored` or specific device name like gpu:1 or None (auto)
    remove_test_compounds: List[str] = [] # remove compounds occuring in the specified (test) datasets
    # data
    isomeric: bool = False      # use isomeric data (if available)
    balance: bool = False       # balance data by dataset
    void_rt: float = 0.0        # void time threshold; used for ALL datasets
    # features
    features: List[str] = []                                     # custom features
    standardize: bool = False                                    # standardize features
    reduce_features: bool = False                                    # standardize features
    num_features: Optional[int] = None
    # additional features
    comp_classes: bool = False  # use classyfire compound classes as add. features
    sysinfo: bool = False       # use column information as add. features
    columns_use_hsm: bool = False
    hsm_fields: List[str] = ['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)']
    custom_column_fields: List[str] = []
    usp_codes: bool = False     # use column usp codes as onehot system features (only for `--sysinfo`)
    debug_onehot_sys: bool = False # onehot dataset encoding
    onehot_test_sets: List[str] = [] # test set IDs to include in onehot encoding
    add_descs: bool = False     # use additional stored descriptors (e.g, qm8)
    classes_l_thr: float = 0.005
    classes_u_thr: float = 0.25
    # model general
    sizes: List[int] = [10, 10] # hidden layer sizes
    dropout_rate: float = 0.0
    # mpn model
    mpn_loss: Literal['margin', 'bce'] = 'margin'
    mpn_margin: float = 0.1
    # pairs
    epsilon: float = 0.5 # difference in evaluation measure below which to ignore falsely predicted pairs
    pair_step: int = 1
    pair_stop: Optional[int] = None
    use_weights: bool = False
    weight_steep: float = 4
    weight_mid: float = 0.75
    # data locations
    repo_root_folder: str = '/home/fleming/Documents/Projects/RtPredTrainingData/'
    add_desc_file: str = '/home/fleming/Documents/Projects/rtranknet/data/qm_merged.csv'
    columns_hsm_data: str = '/home/fleming/Documents/Projects/RtPredTrainingData/resources/hsm_database/hsm_database.txt'
    column_scale_data: str = '/home/fleming/Documents/Projects/rtdata_exploration/data/dataset_info_all.tsv'
    cache_file: str = 'cached_descs.pkl'
    # output control
    verbose: bool = False
    no_progbar: bool = False
    run_name: Optional[str] = None
    export_rois: bool = False
    plot_weights: bool = False
    save_data: bool = False

def generic_run_name():
    from datetime import datetime
    time_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    return f'ranknet_{time_str}'

if __name__ == '__main__':
    args = TrainArgs().parse_args()
    if (args.verbose):
        print(args)
        get_logger().setLevel(INFO)
    else:
        rdBase.DisableLog('rdApp.warning')
    if (args.cache_file is not None):
        features.write_cache = False # flag for reporting changes to cache
        info('reading in cache...')
        if (os.path.exists(args.cache_file)):
            features.cached = pickle.load(open(args.cache_file, 'rb'))
        else:
            features.cached = {}
            info('cache file does not exist yet')
    info('reading in data and computing features...')
    if (args.run_name is None):
        run_name = generic_run_name()
    else:
        run_name = args.run_name
    graphs = (args.model_type == 'mpn')
    # TRAINING
    if (len(args.input) == 1 and os.path.exists(args.input[0])):
        # csv file
        data = Data.from_raw_file(args.input[0], void_rt=args.void_rt)
    elif (all(re.match(r'\d{4}', i) for i in args.input)):
        # dataset IDs (recommended)
        data = Data(use_compound_classes=args.comp_classes, use_system_information=args.sysinfo,
                    classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr,
                    use_usp_codes=args.usp_codes, custom_features=args.features,
                    use_hsm=args.columns_use_hsm, hsm_data=args.columns_hsm_data,
                    column_scale_data=args.column_scale_data,
                    custom_column_fields=args.custom_column_fields or None,
                    hsm_fields=args.hsm_fields, graph_mode=graphs)
        for did in args.input:
            data.add_dataset_id(did,
                                repo_root_folder=args.repo_root_folder,
                                void_rt=args.void_rt,
                                isomeric=args.isomeric)
        if (args.remove_test_compounds is not None and len(args.remove_test_compounds) > 0):
            d_temp = Data()
            for t in args.remove_test_compounds:
                d_temp.add_dataset_id(t, repo_root_folder=args.repo_root_folder,
                                      isomeric=args.isomeric)
            compounds_to_remove = set(d_temp.df['inchi.std'].tolist())
            data.df = data.df.loc[~data.df['inchi.std'].isin(compounds_to_remove)]
            print(f'removed {len(compounds_to_remove)} compounds occuring '
                  'in test data from training data')
        if (args.balance and len(args.input) > 1):
            data.balance()
            info('added data for datasets:\n' +
                 '\n'.join([f'  - {did} ({name})' for did, name in
                            set(data.df[['dataset_id', 'column.name']].itertuples(index=False))]))
    else:
        raise Exception(f'input {args.input} not supported')
    data.compute_features(**parse_feature_spec(args.feature_type), n_thr=args.num_features, verbose=args.verbose,
                          add_descs=args.add_descs, add_desc_file=args.add_desc_file)
    if (args.cache_file is not None and features.write_cache):
        info('writing cache, don\'t interrupt!!')
        pickle.dump(features.cached, open(args.cache_file, 'wb'))
    if args.debug_onehot_sys:
        sorted_dataset_ids = sorted(set(args.input) | set(args.onehot_test_sets))
        data.compute_system_information(True, sorted_dataset_ids)
    info('done. preprocessing...')
    if (graphs):
        data.compute_graphs()
    data.split_data((args.test_split, args.val_split))
    if (args.standardize):
        data.standardize()
    if (args.reduce_features):
        data.reduce_f()
    if (graphs):
        ((train_graphs, train_x, train_y), (val_graphs, val_x, val_y),
         (test_graphs, test_x, test_y)) = data.get_split_data((args.test_split, args.val_split))
        # convert Xs to tensors
        import torch
        info('converting X arrays to torch tensors...')
        train_x = torch.as_tensor(train_x).float()
        val_x = torch.as_tensor(val_x).float()
        test_x = torch.as_tensor(test_x).float()
    else:
        ((train_x, train_y), (val_x, val_y), (test_x, test_y)) = data.get_split_data((args.test_split, args.val_split))
        train_graphs = val_graphs = test_graphs = None
    info('done. Initializing BatchGenerator...')
    bg = BatchGenerator((train_graphs, train_x) if graphs else train_x, train_y,
                        args.batch_size, pair_step=args.pair_step,
                        pair_stop=args.pair_stop, use_weights=args.use_weights,
                        weight_steep=args.weight_steep, weight_mid=args.weight_mid,
                        multix=graphs, y_neg=(args.mpn_loss == 'margin'))
    vg = BatchGenerator((val_graphs, val_x) if graphs else train_x, val_y,
                        args.batch_size, pair_step=args.pair_step,
                        pair_stop=args.pair_stop, use_weights=args.use_weights,
                        weight_steep=args.weight_steep, weight_mid=args.weight_mid,
                        multix=graphs, y_neg=(args.mpn_loss == 'margin'))
    if (args.plot_weights):
        plot_x = np.linspace(0, 10 * args.weight_mid, 100)
        import matplotlib.pyplot as plt
        plt.plot(plot_x, [bg.weight_fn(_, args.weight_steep, args.weight_mid) for _ in plot_x])
        plt.show()
    if (not graphs):
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
        with context:
            v = tf.Variable(1.0)
            info(f'using {v.device}')
            ranker = RankNetNN(input_size=train_x.shape[1],
                               hidden_layer_sizes=args.sizes,
                               activation=(['relu'] * len(args.sizes)),
                               solver='adam',
                               dropout_rate=args.dropout_rate)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                              restore_best_weights=True)
        try:
            ranker.model.fit(bg,
                             callbacks=[es,
                                        # tf.keras.callbacks.TensorBoard(update_freq='epoch', histogram_freq=1,)
                                ],
                             epochs=args.epochs,
                             verbose=1 if not args.no_progbar else 2,
                             validation_data=vg)
        except KeyboardInterrupt:
            print('interrupted training, evaluating...')
        if (args.save_data):
            path = run_name + '.tf'
            ranker.model.save(path, overwrite=True)
            pickle.dump(data, open(os.path.join(path, 'assets', 'data.pkl'), 'wb'))
            json.dump({'train_sets': args.input, 'name': run_name,
                       'args': vars(args)},
                      open(os.path.join(path, 'assets', 'config.json'), 'w'), indent=2)
            print(f'model written to {path}')
        train_preds = predict(train_x, ranker.model, args.batch_size)
        val_preds = predict(val_x, ranker.model, args.batch_size)
        test_preds = predict(test_x, ranker.model, args.batch_size)
    else:
        # MPNranker
        ranker = MPNranker(sigmoid=(args.mpn_loss == 'bce'), extra_features_dim=train_x.shape[1],
                           hidden_units=args.sizes)
        writer = SummaryWriter(f'runs/{run_name}_train')
        val_writer = SummaryWriter(f'runs/{run_name}_val')
        mpn_train(ranker, bg, args.epochs, writer, vg, val_writer=val_writer,
                  steps_train_loss=np.ceil(len(bg) / 100).astype(int),
                  steps_val_loss=np.ceil(len(bg) / 5).astype(int),
                  batch_size=args.batch_size, epsilon=args.epsilon,
                  sigmoid_loss=(args.mpn_loss == 'bce'), margin_loss=args.mpn_margin,
                  early_stopping_patience=args.early_stopping_patience)
        if (args.save_data):
            torch.save(ranker, run_name + '.pt')
            pickle.dump(data, open(os.path.join(f'{run_name}_data.pkl'), 'wb'))
            json.dump({'train_sets': args.input, 'name': run_name,
                       'args': args._log_all()},
                      open(f'{run_name}_config.json', 'w'), indent=2)
        train_preds = mpn_predict((train_graphs, train_x), ranker, batch_size=args.batch_size)
        val_preds = mpn_predict((val_graphs, val_x), ranker, batch_size=args.batch_size)
        test_preds = mpn_predict((test_graphs, test_x), ranker, batch_size=args.batch_size)
    print(f'train: {eval_(train_y, train_preds, args.epsilon):.3f}')
    print(f'test: {eval_(test_y, test_preds, args.epsilon):.3f}')
    print(f'val: {eval_(val_y, val_preds, args.epsilon):.3f}')
    if (False):
        fig = px.treemap(data.df.dropna(subset=['classyfire.kingdom', 'classyfire.superclass', 'classyfire.class']),
                         path=['classyfire.kingdom', 'classyfire.superclass', 'classyfire.class'],
                         title='training data')
        fig.show(renderer='browser')
    if (args.export_rois):
        if not os.path.isdir('runs'):
            os.mkdir('runs')
        export_predictions(data, test_preds, f'runs/{run_name}_test.tsv', 'test')
    if (args.balance and len(args.input) > 1):  # ===LEFT-OUT EVAL===
        # TODO: graphs
        print('evaluating on data left-out when balancing')
        for ds in args.input:
            d = Data(use_compound_classes=args.comp_classes, use_system_information=args.sysinfo,
                     classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr,
                     use_usp_codes=args.usp_codes, custom_features=data.descriptors,
                     use_hsm=args.columns_use_hsm, hsm_data=args.columns_hsm_data,
                     column_scale_data=args.column_scale_data,
                     custom_column_fields=data.custom_column_fields, columns_remove_na=False,
                     hsm_fields=args.hsm_fields)
            d.add_dataset_id(ds,
                             repo_root_folder=args.repo_root_folder,
                             void_rt=args.void_rt,
                             isomeric=args.isomeric)
            perc = len(d.df.loc[d.df.id.isin(data.heldout.id)]) / len(d.df)
            d.df.drop(d.df.loc[~d.df.id.isin(data.heldout.id)].index, inplace=True)
            if (len(d.df) == 0):
                print(f'no data left for {ds}')
                continue
            d.compute_features(mode=parse_feature_spec(args.feature_type), n_thr=args.num_features, verbose=args.verbose,
                               add_descs=args.add_descs, add_desc_file=args.add_desc_file)
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
            preds = predict(X, ranker.model, args.batch_size)
            print(f'{ds}: {eval_(Y, preds, args.epsilon):.3f} \t (#data: {len(Y)}, held-out percentage: {perc:.2f})')
            if (args.export_rois):
                export_predictions(d, preds, f'runs/{run_name}_heldout_{ds}.tsv')
    if (args.cache_file is not None and features.write_cache):
        print('writing cache, don\'t interrupt!!')
        pickle.dump(features.cached, open(args.cache_file, 'wb'))
