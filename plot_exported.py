import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser
import json
from mapping import LADModel
import re

SYMBOLS = {'benchmark_dataset': '▥',
           'our_dataset': '★',
           'structure_disjoint': '⚛',
           'column_disjoint': '❙',
           'setup_disjoint': '⚗'}
BENCHMARK_DATASETS = {'0003': 'IPB_Halle',
                      '0010': 'FEM_lipids',
                      '0018': 'UniToyama_Atlantis',
                      '0055': 'LIFE_new',
                      '0054': 'LIFE_old',
                      '0019': 'Eawag_XBridgeC18',
                      '0002': 'FEM_long'}
# MAEs taken from https://doi.org/10.1016/j.jchromb.2023.123624
BENCHMARK_MAE = {'0003': 0.44,
                 '0010': 1.65,
                 '0018': 1.48,
                 '0055': 0.35,
                 '0054': 0.23,
                 '0019': 0.94,
                 '0002': 2.15}

def make_title(ds, acc=None, lcs_dist=None, flags=None, display_ref_mae=True):
    title = ds
    if ds in BENCHMARK_DATASETS:
        title += f' "{BENCHMARK_DATASETS[ds]}"'
    if acc is not None or lcs_dist is not None:
        title += ' '
        acc = f'{acc:.1%}' if acc is not None else ''
        # lcs_dist = r'$d_{LCS}=' + f'{lcs_dist:.1f}' + r'$' if lcs_dist is not None else ''
        lcs_dist = r'LCS=' + f'{lcs_dist:.0f}' + r'' if lcs_dist is not None else ''
        title += '(' + ', '.join([acc, lcs_dist]) + ')'
    if flags is not None:
        title += '  '
        for desc, symbol in SYMBOLS.items():
            if desc in flags and flags[desc]:
                title += symbol
    if display_ref_mae and ds in BENCHMARK_MAE:
        title += f'\nreference MAE: {BENCHMARK_MAE[ds]:.2f}'
    return title

def sort_flags(flags):
    if not any(f in SYMBOLS.values() for f in flags):
        return np.infty
    # return most important symbol (min) in flags
    flag_order = {SYMBOLS[k]: i for i, k in enumerate([
        'benchmark_dataset', 'column_disjoint', 'setup_disjoint', 'structure_disjoint', 'our_dataset'])}
    res = min([flag_order[f] for f in flags])
    return res

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('tsvs', nargs='+')
    parser.add_argument('--out', default=None)
    parser.add_argument('--out_errors', default=None)
    parser.add_argument('--accs_file', default=None)
    parser.add_argument('--errorlabels', action='store_true')
    parser.add_argument('--dont_show', action='store_true')
    parser.add_argument('--repo_root', default='/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/')
    parser.add_argument('--void_factor', default=2, type=float)
    parser.add_argument('--extrap', default=None, type=int)
    parser.add_argument('--anchors', default=None, type=int)
    parser.add_argument('--onlybest', action='store_true')
    parser.add_argument('--showolsfit', action='store_true')
    parser.add_argument('--no_negative_ols', action='store_true')
    parser.add_argument('--ols_drop_mode', choices=['50%', '2*median'], default='50%')
    parser.add_argument('--no_sort_flags', action='store_true')
    parser.add_argument('--extra_bases', action='store_true')
    parser.add_argument('--extra_bases_nr_thr', default=-1, type=int)
    # parser.add_argument('--bases', nargs='+', default=['1', 'x', 'x**2'], type=str, help='supported are 1, x, x**2, sqrt(x), x*sqrt(x)')
    args = parser.parse_args()
    # args = parser.parse_args('/home/fleming/Documents/Projects/rtranknet/runs/FEbenchmark/FEbenchmark_reporthsmtanakaph_benchmarkpartly/FEbenchmark_reporthsmtanakaph_benchmarkpartly_0002_train.tsv.tsv /home/fleming/Documents/Projects/rtranknet/runs/FEbenchmark/FEbenchmark_reporthsmtanakaph_benchmarkpartly/FEbenchmark_reporthsmtanakaph_benchmarkpartly_0002_test.tsv.tsv --repo_root /home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/ --errorlabels --anchor 15'.split())
    sys.path.append(args.repo_root)
    from pandas_dfs import get_dataset_df
    dss = get_dataset_df()
    models = defaultdict(dict)
    roi_cutoff = defaultdict(dict)
    data = {}
    normal_bases = ['1', 'x', 'x**2']
    # load data + make LAD models
    input_data = {}
    for f in args.tsvs:
        ds_id, split = re.match(r'.*_(\d{4})(_test|_train)?.tsv', f.split('/')[-1]).groups()
        # TODO: won't work with non-RepoRT datasets
        df = pd.read_csv(f, sep='\t', names=['smiles', 'rt', 'roi'], header=None)
        if split is not None:
            # there is a train and a test portion
            df['split'] = split.strip('_')
            if ds_id in input_data:
                combined = pd.concat([input_data[ds_id], df])
                input_data[ds_id] = combined
            else:
                input_data[ds_id] = df
    for id_, df in input_data.items():
        print(f'{id_}: ROI->RT modeling')
        df['roi2'] = df.roi ** 2 # for LAD model
        data[id_] = df.copy()
        if 'split' in df.columns and (df.split == 'train').sum() > 0:
            # only use the train portion for the model
            df = df.loc[df.split == 'train']
            print(f'{id_}: building LAD models on the train portion (max #compounds {len(df)}, full dataset {len(data[id_])})')
        # models['all_points'][ds] = LADModel(data[ds])
        void_t = dss['column.t0'].loc[id_] * args.void_factor
        if (not args.onlybest):
            models['LAD'][id_] = LADModel(df, void=void_t, ols_after=False)
        models['LAD+OLS'][id_] = LADModel(df, void=void_t, ols_after=True,
                                          ols_discard_if_negative=args.no_negative_ols,
                                          ols_drop_mode=args.ols_drop_mode)
        if args.extrap is not None:
            extrap = f'extrap{args.extrap}'
            data_void = df.loc[df.rt > void_t]
            roi_cutoff[extrap][id_] = data_void.roi.sort_values().iloc[int(len(data_void) * (args.extrap / 100))]
            models[extrap][id_] = LADModel(df.loc[df.roi < roi_cutoff[extrap][id_]], void=void_t,
                                           ols_after=True, ols_discard_if_negative=args.no_negative_ols,
                                           ols_drop_mode=args.ols_drop_mode)
        if args.anchors is not None:
            anchors = f'anchors{args.anchors}'
            data_void = df.loc[df.rt > void_t]
            data_anchors = data_void.sample(args.anchors)
            models[anchors][id_] = LADModel(data_anchors, void=void_t,
                                           ols_after=True, ols_discard_if_negative=args.no_negative_ols,
                                           ols_drop_mode=args.ols_drop_mode)
            models[anchors][id_].anchor_points = data_anchors
            models[anchors][id_].anchor_points['rt_pred'] = models[anchors][id_].get_mapping(data_anchors.roi)
        if args.extra_bases and (args.extra_bases_nr_thr == -1 or
                                 len(df.loc[df.rt > void_t]) >= args.extra_bases_nr_thr):
                models['LAD+OLS (extra)'][id_] = LADModel(df, void=void_t, ols_after=True,
                                                          ols_discard_if_negative=args.no_negative_ols,
                                                          bases=normal_bases + ['sqrt(x)', 'x*sqrt(x)'],
                                                          ols_drop_mode=args.ols_drop_mode)
    ds_list = list(data)
    # optionally various other information
    accs = defaultdict(lambda: None)
    lcs_dists = defaultdict(lambda: None)
    flags = defaultdict(lambda: None)
    if (args.accs_file is not None):
        for k, v in json.load(open(args.accs_file)).items():
            if 'lcs_dist' in v:
                lcs_dists[k] = v['lcs_dist']
            if 'acc' in v:
                accs[k] = v['acc']
            if 'flags' in v:
                flags[k] = v['flags']
    # make titles to use for sorting datasets
    titles = {ds: make_title(ds, acc=accs[ds], lcs_dist=lcs_dists[ds], flags=flags[ds])
              for ds in ds_list}
    # plot
    figsize = (19, np.ceil(len(ds_list) / 4) * 4.5)
    fig,axes = plt.subplots(nrows=np.ceil(len(ds_list) / 4).astype(int), ncols=4,
                             figsize=figsize)
    #fig.set_dpi(150)
    errors = []
    for ds, ax in zip(sorted(ds_list, key=lambda x: (sort_flags(titles[x].split('\n')[0].split()[-1]) if not args.no_sort_flags else None,
                                                int(x))),
                      axes.ravel()):
        colors = plt.rcParams["axes.prop_cycle"]()
        x = np.arange(data[ds].roi.min(), data[ds].roi.max(), 0.001)
        # TODO: maybe density, commented out code works in principle
        # data[ds]['density'] = gaussian_kde(np.vstack([data[ds].rt, data[ds].roi]))(
        #     np.vstack([data[ds].rt, data[ds].roi]))
        # sns.scatterplot(data=data[ds], x='roi', y='rt', hue='density', s=2, ax=ax)
        ax.scatter(data[ds].roi, data[ds].rt, s=2)
        for type_ in models:
            if ds not in models[type_]:
                continue        # some models are only made for certain datasets
            c = next(colors)['color']
            model = models[type_][ds]
            y = model.get_mapping(x)
            data_rel = data[ds].loc[data[ds].rt >= dss['column.t0'].loc[ds] * args.void_factor]
            split_errors = {}
            if 'split' in data_rel.columns and (data_rel.split == 'test').sum() > 0:
                # only use the test portion for the error
                test_df = data_rel.loc[data_rel.split == 'test']
                train_df = data_rel.loc[data_rel.split == 'train']
                error_test = (model.get_mapping(test_df.roi) - test_df.rt).abs()
                error_train = (model.get_mapping(train_df.roi) - train_df.rt).abs()
                print(f'{ds} train error: MAE={error_train.mean():.2f}, MedAE={error_train.median():.2f})')
                print(f'{ds} test error: MAE={error_test.mean():.2f}, MedAE={error_test.median():.2f})')
                error = error_test
                split_errors = dict(train=error_train, test=error_test)
            else:
                error = (model.get_mapping(data_rel.roi) - data_rel.rt).abs()
            match model.no_ols_why:
                case 'NEGATIVE_COEFFICIENTS':
                    description = f'{type_.replace("+OLS", "")} (OLS noninc)'
                case 'OLS_FAILED':
                    description = f'{type_.replace("+OLS", "")} (OLS failed)'
                case _:
                    description = type_
            errors_record = {'model_type': description, 'ds': ds, 'MAE': error.mean(), 'MedAE': error.median()}
            for k, v in split_errors.items():
                errors_record[f'MAE_{k}'] = v.mean()
                errors_record[f'MedAE_{k}'] = v.median()
            errors.append(errors_record)
            ax.plot(x, y, color=c,
                    label=f'{description}' + (f'(MAE={error.mean():.2f}, MedAE={error.median():.2f})'
                                        if args.errorlabels else ''))
            if (args.showolsfit and not 'extrap' in type_ and hasattr(model, 'ols_points')):
                ax.scatter(model.ols_points.roi, model.ols_points.rt_pred,
                           label='OLS fit points', c=c, s=5)
            if (hasattr(model, 'anchor_points')):
                ax.scatter(model.anchor_points.roi, model.anchor_points.rt_pred,
                           label='anchor points', c=c, s=5)
        ax.set_title(titles[ds])
        ax.set_xlabel('ROI')
        ax.set_ylabel('rt (min)')
        # ax.axhline(dss['column.t0'].loc[ds], linestyle='dotted', color='red')
        ax.axhline(dss['column.t0'].loc[ds] * args.void_factor, linestyle='dotted', color='red', label='void cutoff')
        # ax.axhline(dss['column.t0'].loc[ds] * 3, linestyle='dotted', color='green')
        ax.legend()
    plt.tight_layout()
    # ax.set_xlim((240, 3000))
    # ax.set_ylim((240, 2000))
    # legend = ax.legend()
    # ax.set_xlabel('Income', fontsize=16)
    # ax.set_ylabel('Food expenditure', fontsize=16);
    if (args.out is not None):
        plt.savefig(args.out)
    errors_df = pd.DataFrame.from_records(errors).set_index(['ds', 'model_type'])
    print(errors_df)
    if (args.out_errors is not None):
        errors_df.to_csv(args.out_errors)
    if (not args.dont_show):
        plt.show()
