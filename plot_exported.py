from evaluate import visualize_df
import pandas as pd
import seaborn as sns
from glob import glob
import statsmodels.api as sm
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sys
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, getSolver
from argparse import ArgumentParser
import json

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
parser.add_argument('--onlybest', action='store_true')
parser.add_argument('--showolsfit', action='store_true')

class LADModel:
    def __init__(self, data, void=0, ols_after=False):
        model = LpProblem(name='LAD', sense=LpMinimize)
        if (void > 0):
            data = data.loc[data.rt > void]
        x = data.roi.values
        y = data.rt.values
        n = len(x)
        u = [LpVariable(name=f'u{i}') for i in range(n)]
        a = LpVariable(name='a', lowBound=0)
        b = LpVariable(name='b', lowBound=0)
        c = LpVariable(name='c', lowBound=0)
        for i in range(n):
            model += u[i] >= y[i] - a * x[i] ** 2 - b * x[i] - c
            model += u[i] >= - (y[i] - a * x[i] ** 2 - b * x[i] - c)
        model += lpSum(u)
        status = model.solve(getSolver('PULP_CBC_CMD', msg=False))
        # status = model.solve()
        assert status == 1, 'solution not optimal'
        self.get_y = lambda x: a.varValue * x ** 2 + b.varValue * x + c.varValue
        if (ols_after):
            self.data_keep = np.argsort(np.abs([self.get_y(xi) for xi in x] - y))[:len(x) // 2]
            x = x[self.data_keep]
            y = y[self.data_keep]
            self.ols_points = (x, y)
            import statsmodels.api as sm
            X = sm.add_constant(np.array([x, x**2]).transpose())
            model = sm.OLS(y, X)
            try:
                const, x1, x2 = model.fit().params # TODO: with too few data points: error
                self.get_y = lambda x: x2 * x ** 2 + x1 * x + const
            except:
                print('not enough data points for OLS model')

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
    args = parser.parse_args()
    sys.path.append(args.repo_root)
    from pandas_dfs import get_dataset_df
    dss = get_dataset_df()
    models = defaultdict(dict)
    roi_cutoff = defaultdict(dict)
    data = {}
    # load data + make LAD models
    for ds in args.tsvs:
        id_ = ds.split('.')[-2].split('_')[-1]
        print(f'{id_}: ROI->RT modeling')
        df = pd.read_csv(ds, sep='\t', names=['smiles', 'rt', 'roi'], header=None)
        df['roi2'] = df.roi ** 2 # for LAD model
        data[id_] = df
        # models['all_points'][ds] = LADModel(data[ds])
        void_t = dss['column.t0'].loc[id_] * args.void_factor
        if (not args.onlybest):
            models['LAD'][id_] = LADModel(data[id_], void=void_t)
        models['LAD+OLS'][id_] = LADModel(data[id_], void=void_t, ols_after=True)
        if args.extrap is not None:
            extrap = f'extrap{args.extrap}'
            data_void = data[id_].loc[data[id_].rt > void_t]
            roi_cutoff[extrap][id_] = data_void.roi.sort_values().iloc[int(len(data_void) * (args.extrap / 100))]
            models[extrap][id_] = LADModel(data[id_].loc[data[id_].roi < roi_cutoff[extrap][id_]], void=void_t, ols_after=True)
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
    for ds, ax in zip(sorted(ds_list, key=lambda x: (sort_flags(titles[x].split('\n')[0].split()[-1]), int(x))),
                      axes.ravel()):
        colors = plt.rcParams["axes.prop_cycle"]()
        x = np.arange(data[ds].roi.min(), data[ds].roi.max(), 0.001)
        # TODO: maybe density, commented out code works in principle
        # data[ds]['density'] = gaussian_kde(np.vstack([data[ds].rt, data[ds].roi]))(
        #     np.vstack([data[ds].rt, data[ds].roi]))
        # sns.scatterplot(data=data[ds], x='roi', y='rt', hue='density', s=2, ax=ax)
        ax.scatter(data[ds].roi, data[ds].rt, s=2)
        for type_ in models:
            c = next(colors)['color']
            y = models[type_][ds].get_y(x)
            data_rel = data[ds].loc[data[ds].rt >= dss['column.t0'].loc[ds] * args.void_factor]
            error = (models[type_][ds].get_y(data_rel.roi) - data_rel.rt).abs()
            errors.append({'model_type': type_, 'ds': ds, 'MAE': error.mean(), 'MedAE': error.median()})
            ax.plot(x, y, color=c,
                    label=f'{type_}' + (f'(MAE={error.mean():.2f}, MedAE={error.median():.2f})'
                                        if args.errorlabels else ''))
            if (args.showolsfit and not 'extrap' in type_ and hasattr(models[type_][ds], 'ols_points')):
                ax.scatter(models[type_][ds].ols_points[0], models[type_][ds].ols_points[1],
                           label='OLS fit points', c=c, s=5)
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
