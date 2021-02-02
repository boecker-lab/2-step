import numpy as np
import tensorflow as tf
from LambdaRankNN import RankNetNN
from rdkit import rdBase
import pickle
import json
import os
import argparse
import re
import contextlib
from itertools import product

from utils import BatchGenerator, Data
from features import features, parse_feature_spec
from evaluate import eval_, predict, export_predictions


def generic_run_name():
    from datetime import datetime
    time_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    return f'ranknet_{time_str}'

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Retention Order index prediction')
    parser.add_argument('-i', '--input', default=['example.csv'], help='Either CSV or dataset IDs',
                        nargs='+')
    parser.add_argument('-t', '--type', help='type of features',
                        default='rdk',
                        choices=['rdkall', 'rdk2d', 'rdk3d']
                        + [f'ae{i}{j}' for i, j in product(range(2), range(3))])
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
    parser.add_argument('--add_descs', action='store_true', help='use additional stored descriptors (e.g, qm8)')
    parser.add_argument('--repo_root_folder', default='/home/fleming/Documents/Projects/RtPredTrainingData/',
                        help='location of the dataset github repository')
    parser.add_argument('--add_desc_file', default='/home/fleming/Documents/Projects/RtPredTrainingData/',
                        help='csv with additional features with smiles as identifier')
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
    parser.add_argument('--test_split', default=0.2, type=float, help=' ')
    parser.add_argument('--val_split', default=0.05, type=float, help=' ')
    parser.add_argument('--use_weights', action='store_true', help=' ')
    parser.add_argument('--weight_steep', default=4, type=float, help=' ')
    parser.add_argument('--weight_mid', default=0.75, type=float, help=' ')
    parser.add_argument('--plot_weights', action='store_true', help=' ')
    parser.add_argument('--debug_onehot_sys', action='store_true', help=' ')
    parser.add_argument('--device', default=None,
                        help='can be `mirrored`, a specific device name like `gpu:1` '
                        'or `None` which automatically selects an option')
    parser.add_argument('--epsilon', type=float, default=1.,
                        help='difference in evaluation measure below which to ignore falsely predicted pairs')
    parser.add_argument('--columns_use_hsm', action='store_true', help=' ')
    parser.add_argument('--columns_hsm_data', default=
                        '/home/fleming/Documents/Projects/RtPredTrainingData/resources/hsm_database/hsm_database.txt',
                        help=' ')
    parser.add_argument('--custom_column_fields', default=None,
                        nargs='*', help=' ')
    parser.add_argument('--hsm_fields', default=['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)'],
                        nargs='*', help=' ')
    parser.add_argument('--remove_test_compounds', nargs='+',
                        help='remove compounds occuring in the specified test sets')
    parser.add_argument('--save', action='store_true', help=' ')
    return parser.parse_args() if args is None else parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arguments()
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
    if (args.run_name is None):
        run_name = generic_run_name()
    else:
        run_name = args.run_name
    # TRAINING
    if (len(args.input) == 1 and os.path.exists(args.input[0])):
        # csv file
        data = Data.from_raw_file(args.input[0], void_rt=args.void_rt)
    elif (all(re.match(r'\d{4}', i) for i in args.input)):
        # dataset IDs (recommended)
        data = Data(use_compound_classes=args.cclasses, use_system_information=args.sysinfo,
                    classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr,
                    use_usp_codes=args.usp_codes, custom_features=args.features,
                    use_hsm=args.columns_use_hsm, hsm_data=args.columns_hsm_data,
                    custom_column_fields=args.custom_column_fields,
                    hsm_fields=args.hsm_fields)
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
        if (args.verbose):
            print('added data for datasets:')
            print('\n'.join([f'  - {did} ({name})' for did, name in
                             set(data.df[['dataset_id', 'column.name']].itertuples(index=False))]))
    else:
        raise Exception(f'input {args.input} not supported')
    data.compute_features(**parse_feature_spec(args.type), n_thr=args.num_features, verbose=args.verbose,
                          add_descs=args.add_descs)
    if (args.cache_file is not None and features.write_cache):
        if (args.verbose):
            print('writing cache, don\'t interrupt!!')
        pickle.dump(features.cached, open(args.cache_file, 'wb'))
    if args.debug_onehot_sys:
        sorted_dataset_ids = sorted(set(args.input) | set(args.test))
        data.compute_system_information(True, sorted_dataset_ids)
    if (args.verbose):
        print('done. preprocessing...')
    data.split_data((args.test_split, args.val_split))
    if (args.standardize):
        data.standardize()
    if (args.reduce_features):
        data.reduce_f()
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = data.get_split_data()
    if (args.verbose):
        print('done. Initializing BatchGenerator...')
    bg = BatchGenerator(train_x, train_y, args.batch_size, pair_step=args.pair_step,
                        pair_stop=args.pair_stop, use_weights=args.use_weights,
                        weight_steep=args.weight_steep, weight_mid=args.weight_mid)
    if (args.plot_weights):
        plot_x = np.linspace(0, 10 * args.weight_mid, 100)
        import matplotlib.pyplot as plt
        plt.plot(plot_x, [bg.weight_fn(_, args.weight_steep, args.weight_mid) for _ in plot_x])
        plt.show()
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
                             pair_stop=args.pair_stop, use_weights=args.use_weights,
                             weight_steep=args.weight_steep, weight_mid=args.weight_mid))
    except KeyboardInterrupt:
        print('interrupted training, evaluating...')
    if (args.save):
        path = run_name + '.tf'
        ranker.model.save(path, overwrite=True)
        pickle.dump(data, open(os.path.join(path, 'assets', 'data.pkl'), 'wb'))
        json.dump({'train_sets': args.input, 'name': run_name,
                   'args': vars(args)},
                  open(os.path.join(path, 'assets', 'config.json'), 'w'), indent=2)
        print(f'model written to {path}')
    print(f'train: {eval_(train_y, predict(train_x, ranker.model, args.batch_size), args.epsilon):.3f}')
    test_preds = predict(test_x, ranker.model, args.batch_size)
    print(f'test: {eval_(test_y, test_preds, args.epsilon):.3f}')
    print(f'val: {eval_(val_y, predict(val_x, ranker.model, args.batch_size), args.epsilon):.3f}')
    if (False and args.classyfire):
        fig = px.treemap(data.df.dropna(subset=['classyfire.kingdom', 'classyfire.superclass', 'classyfire.class']),
                         path=['classyfire.kingdom', 'classyfire.superclass', 'classyfire.class'],
                         title='training data')
        fig.show(renderer='browser')
    if (args.export_rois):
        if not os.path.isdir('runs'):
            os.mkdir('runs')
        export_predictions(data, test_preds, f'runs/{run_name}_test.tsv', 'test')
    if (args.balance and len(args.input) > 1):  # ===LEFT-OUT EVAL===
        print('evaluating on data left-out when balancing')
        for ds in args.input:
            d = Data(use_compound_classes=args.cclasses, use_system_information=args.sysinfo,
                     classes_l_thr=args.classes_l_thr, classes_u_thr=args.classes_u_thr,
                     use_usp_codes=args.usp_codes, custom_features=data.descriptors,
                     use_hsm=args.columns_use_hsm, hsm_data=args.columns_hsm_data,
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
            d.compute_features(mode=parse_feature_spec(args.type), n_thr=args.num_features, verbose=args.verbose)
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
        if (args.verbose):
            print('writing cache, don\'t interrupt!!')
        pickle.dump(features.cached, open(args.cache_file, 'wb'))
