import pandas as pd
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser
import json
import yaml
from mapping import LADModel
import re
from os.path import basename

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

def make_title(ds, acc=None, mcd_ratio=None, flags=None, display_ref_mae=False):
    title = ds[1] if isinstance(ds, tuple) else ds
    if ds in BENCHMARK_DATASETS:
        title += f' "{BENCHMARK_DATASETS[ds]}"'
    if acc is not None or mcd_ratio is not None:
        title += ' '
        acc = f'{acc:.1%}' if acc is not None else ''
        mcd_ratio = r'MCD=' + f'{mcd_ratio:.2f}' + r'' if mcd_ratio is not None else ''
        title += '(' + ', '.join([acc, mcd_ratio]) + ')'
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
        return np.inf
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
    parser.add_argument('--repo_root', default='../RepoRT/')
    parser.add_argument('--void_factor', default=2, type=float)
    parser.add_argument('--hide_void', action='store_true', help='don\'t show data in void volume in plot')
    parser.add_argument('--overwrite_void_thresholds', default=None, help='yaml file containing estimations with dataset IDs as key')
    parser.add_argument('--extrap', default=None, type=int)
    parser.add_argument('--anchors', default=None, type=str)
    parser.add_argument('--anchors_only', action='store_true')
    parser.add_argument('--onlybest', action='store_true')
    parser.add_argument('--showolsfit', action='store_true')
    parser.add_argument('--no_negative_ols', action='store_true')
    parser.add_argument('--ols_drop_mode', choices=['50%', '2*median'], default='50%')
    parser.add_argument('--no_sort_flags', action='store_true')
    parser.add_argument('--no_sort_datasets', action='store_true')
    parser.add_argument('--basename_as_id', action='store_true')
    parser.add_argument('--extra_bases', action='store_true')
    parser.add_argument('--extra_bases_nr_thr', default=-1, type=int)
    parser.add_argument('--ncols', default=4, type=int)
    parser.add_argument('--no_legend', action='store_true')
    parser.add_argument('--error_in_titles', action='store_true')
    parser.add_argument('--extra_title_info', default=None, type=str)
    parser.add_argument('--plot_anchor_points', action='store_true')
    parser.add_argument('--normal_mapping_anchors_perc', default=90, type=int)
    # parser.add_argument('--bases', nargs='+', default=['1', 'x', 'x**2'], type=str, help='supported are 1, x, x**2, sqrt(x), x*sqrt(x)')
    args = parser.parse_args()
    sys.path.append(args.repo_root)
    from pandas_dfs import get_dataset_df
    dss = get_dataset_df()
    models = defaultdict(dict)
    roi_cutoff = defaultdict(dict)
    data = OrderedDict()
    normal_bases = ['1', 'x', 'x**2']
    # load data + make LAD models
    input_data = {}
    void_thresholds = {}
    if (args.overwrite_void_thresholds is not None):
        void_thresholds |= yaml.load(open(args.overwrite_void_thresholds), yaml.loader.SafeLoader)
    for f in args.tsvs:
        ds_id, fold, split = re.match(r'.*_(\d{4})(_fold\d+)?(_test|_train)?.tsv', f.split('/')[-1]).groups()
        # TODO: won't work with non-RepoRT datasets
        df = pd.read_csv(f, sep='\t', names=['smiles', 'rt', 'roi'], header=None)
        data_dict_id = ds_id if (not args.basename_as_id) else (basename(f), ds_id)
        if split is not None:
            # there is a train and a test portion
            df['split'] = split.strip('_')
            if data_dict_id in input_data:
                combined = pd.concat([input_data[data_dict_id], df])
                input_data[data_dict_id] = combined
            else:
                input_data[data_dict_id] = df
        else:
            input_data[data_dict_id] = df
    for id_, df in input_data.items():
        report_id = id_[1] if isinstance(id_, tuple) else id_
        print(f'{id_}: ROI->RT modeling')
        df['roi2'] = df.roi ** 2 # for LAD model
        data[id_] = df.copy()
        if 'split' in df.columns and (df.split == 'train').sum() > 0:
            # only use the train portion for the model
            df = df.loc[df.split == 'train']
            print(f'{id_}: building LAD models on the train portion (max #compounds {len(df)}, full dataset {len(data[id_])})')
        # models['all_points'][ds] = LADModel(data[ds])
        if (report_id not in void_thresholds):
            void_thresholds[report_id] = dss['column.t0'].loc[report_id] * args.void_factor
        void_t = void_thresholds[report_id]
        if (not args.anchors_only):
            data_void = df.loc[df.rt > void_t]
            data_mapping = data_void.sample(np.round(len(data_void) * (args.normal_mapping_anchors_perc / 100)).astype(int))
            if (not args.onlybest):
                models['LAD'][id_] = LADModel(data_mapping, void=void_t, ols_after=False)
            models['LAD+OLS'][id_] = LADModel(data_mapping, void=void_t, ols_after=True,
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
            if args.anchors.endswith('%'):
                # percentage
                data_anchors = data_void.sample(np.round(len(data_void) * (float(args.anchors.replace('%', '')) / 100)).astype(int))
                print(f'{id_}: using {len(data_anchors)} ({args.anchors}) out of {len(data_void)}({len(df)}) data points for the mapping')
            else:
                # absolute number
                data_anchors = data_void.sample(int(args.anchors))
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
    mcd_ratios = defaultdict(lambda: None)
    flags = defaultdict(lambda: None)
    if (args.accs_file is not None):
        for k, v in json.load(open(args.accs_file)).items():
            if 'mcd_ratio' in v:
                mcd_ratios[k] = v['mcd_ratio']
            if 'acc' in v:
                accs[k] = v['acc']
            if 'flags' in v:
                flags[k] = v['flags']
    # make titles to use for sorting datasets
    titles = {ds: make_title(ds, acc=accs[ds], mcd_ratio=mcd_ratios[ds], flags=flags[ds])
              for ds in ds_list}
    # plot
    figsize = (4.5 * args.ncols, np.ceil(len(ds_list) / args.ncols) * 4.5)
    # TODO: make sure figsize leads to exactly equal size ratios for each subplot
    fig,axes = plt.subplots(nrows=np.ceil(len(ds_list) / args.ncols).astype(int), ncols=args.ncols,
                             figsize=figsize)
    #fig.set_dpi(150)
    errors = []
    if args.no_sort_datasets:
        ds_iter = ds_list
    else:
        ds_iter = sorted(ds_list, key=lambda x: (sort_flags(titles[x].split('\n')[0].split()[-1]) if not args.no_sort_flags else None,
                                            int(x)))
    for ds, ax in zip(ds_iter,
                      axes.ravel()):
        report_id = ds[1] if isinstance(ds, tuple) else ds
        colors = plt.rcParams["axes.prop_cycle"]()
        data_rel = data[ds].loc[data[ds].rt >= void_thresholds[report_id]]
        if (args.hide_void):
            x = np.arange(data_rel.roi.min(), data_rel.roi.max(), 0.001)
            # ax.scatter(data_rel.roi, data_rel.rt, c='grey', s=10)
            ax.scatter(data_rel.roi, data_rel.rt, c='grey', s=5)
        else:
            x = np.arange(data[ds].roi.min(), data[ds].roi.max(), 0.001)
            ax.scatter(data[ds].roi, data[ds].rt, c='grey', s=5)
        # TODO: maybe density, commented out code works in principle
        # data[ds]['density'] = gaussian_kde(np.vstack([data[ds].rt, data[ds].roi]))(
        #     np.vstack([data[ds].rt, data[ds].roi]))
        # sns.scatterplot(data=data[ds], x='roi', y='rt', hue='density', s=2, ax=ax)
        title_extra_info = ' '
        for type_ in models:
            if ds not in models[type_]:
                continue        # some models are only made for certain datasets
            c = next(colors)['color']
            # c = '#ff7f0e'
            model = models[type_][ds]
            y = model.get_mapping(x)
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
            ax.plot(x, y, color=c, linewidth=2.5,
                    label=f'{description}' + (f'(MAE={error.mean():.2f}, MedAE={error.median():.2f})'
                                              if args.errorlabels else ''))
            if (args.showolsfit and not 'extrap' in type_ and hasattr(model, 'ols_points')):
                ax.scatter(model.ols_points.roi, model.ols_points.rt_pred,
                           label='OLS fit points', c=c)
            if (args.plot_anchor_points and hasattr(model, 'anchor_points')):
                ax.scatter(model.anchor_points.roi, model.anchor_points.rt_pred,
                           label='Anchors', c='black', s=10)
            if (args.error_in_titles):
                title_extra_info += f'(MAE={error.mean():.2f} min)'
            if (args.extra_title_info is not None):
                title_extra_info = title_extra_info.rstrip(')') + f' {args.extra_title_info})'
        ax.set_title((titles[ds] + title_extra_info).strip())
        ax.set_xlabel('Retention Order Index')
        ax.set_ylabel('Retention time (min)')
        if (not args.hide_void):
            # ax.axhline(dss['column.t0'].loc[report_id], linestyle='dotted', color='red')
            ax.axhline(void_thresholds[report_id], linestyle='dotted', color='red', label='Void volume threshold', linewidth=2.)
            # ax.axhline(dss['column.t0'].loc[report_id] * 3, linestyle='dotted', color='green')
            # TODO: debug plot original void lines
            # ax.axhline(dss['column.t0'].loc[report_id] * 2, linestyle='dotted', color='orange', label='Void volume threshold original', linewidth=2.)
        if (not args.no_legend):
            ax.legend()
    plt.tight_layout()
    # ax.set_xlim((240, 3000))
    # ax.set_ylim((240, 2000))
    # legend = ax.legend()
    # ax.set_xlabel('Income', fontsize=16)
    # ax.set_ylabel('Food expenditure', fontsize=16);
    if (args.out is not None):
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(args.out)
    errors_df = pd.DataFrame.from_records(errors).set_index(['ds', 'model_type'])
    print(errors_df)
    if (args.out_errors is not None):
        errors_df.to_csv(args.out_errors)
    if (not args.dont_show):
        plt.show()
