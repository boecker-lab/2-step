"""
computes mapping between ROI and RT using Least Absolute Deviation (LAD) regression.
By default, LAD coefficients are enforced to be non-negative.
Optionally, mapping can be improved afterwards using Ordinary Least Squares (OLS).

With `ols_discard_if_negative`, coefficients from OLS will only be kept if non-negative:
>>> LADModel(ols_neg_data, ols_after=True, ols_discard_if_negative=True).no_ols_why
LAD coefficients: 3.8, 15.1, 0.0
OLS coefficients: 2.6, 24.2, -9.3
final coefficients: 3.8, 15.1, 0.0
'NEGATIVE_COEFFICIENTS'

>>> LADModel(ols_neg_data, ols_after=True, ols_discard_if_negative=False).no_ols_why
LAD coefficients: 3.8, 15.1, 0.0
OLS coefficients: 2.6, 24.2, -9.3
final coefficients: 2.6, 24.2, -9.3

OLS can also fail with too few data points
>>> LADModel(mini_data, ols_after=True, ols_discard_if_negative=True, verbose=False).no_ols_why
'OLS_FAILED'
"""

from typing import Literal
import numpy as np
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, getSolver
from logging import warning

class LADModel:
    def __init__(self, data, void=0, ols_after=False, ols_discard_if_negative=False,
                 ols_drop_mode:Literal['50%', '2*median']='2*median', bases=['1', 'x', 'x**2'],
                 verbose=True):
        self.data_input = data
        self.void = void
        self.ols_after = ols_after
        self.ols_discard_if_negative = ols_discard_if_negative
        self.ols_drop_mode = ols_drop_mode
        self.data = data.copy() if void == 0 else data.loc[data.rt > void]
        self.bases = bases
        self.verbose = verbose
        self._compute()

    def _compute(self):
        self.lad_coefficients = self._compute_lad_coefficients(self.data)
        if (self.verbose):
            print('LAD coefficients:', ', '.join(f'{c:.1f}' for c in self.lad_coefficients))
        self.coefficients = self.lad_coefficients
        self.no_ols = not self.ols_after
        self.no_ols_why = None
        if (self.ols_after):
            ols_data = self.data.copy()
            ols_data['rt_pred'] = self.get_mapping(ols_data.roi)
            ols_data['MAE'] = (ols_data['rt_pred'] - ols_data.rt).abs()
            pre_drop_len = len(ols_data)
            if (self.ols_drop_mode == '50%'):
                ols_data = ols_data.sort_values('MAE').iloc[:(len(ols_data) // 2)]
            elif (self.ols_drop_mode == '2*median'):
                median_error = ols_data.MAE.median()
                ols_data = ols_data.loc[ols_data.MAE <= 2 * median_error]
            print(f'dropped {pre_drop_len - len(ols_data)} data points for OLS [{self.ols_drop_mode}]')
            self.ols_data = ols_data
            self.ols_coefficients = self._get_ols_refined_coefficients(ols_data)
            if self.ols_coefficients is not None:
                if (self.verbose):
                    print('OLS coefficients:', ', '.join(f'{c:.1f}' for c in self.ols_coefficients))
                if (any([coefficient < 0 for coefficient in self.ols_coefficients])):
                    warning('Applying the OLS leads to negative coefficients: '
                            ', '.join(f'{c:.1f}({c_orig:.1f})' for c, c_orig in zip(self.ols_coefficients, self.lad_coefficients)))
                    if (self.ols_discard_if_negative):
                        self.no_ols = True
                        self.no_ols_why = 'NEGATIVE_COEFFICIENTS'
            else:
                warning('OLS model failed, keeping LAD coefficients')
                self.no_ols = True
                self.no_ols_why = 'OLS_FAILED'
            if (not self.no_ols):
                self.coefficients = self.ols_coefficients
        if (self.verbose):
            print('final coefficients:', ', '.join(f'{c:.1f}' for c in self.coefficients))

    def _compute_lad_coefficients(self, data, enforce_positive=True):
        model = LpProblem(name='LAD', sense=LpMinimize)
        x = data.roi.values
        y = data.rt.values
        n = len(x)
        u = [LpVariable(name=f'u{i}') for i in range(n)]
        var_args = dict(lowBound=0) if enforce_positive else dict()
        coefficients = [LpVariable(name=f'c{k}', **var_args) for k in range(len(self.bases))]
        for i in range(n):
            model += u[i] >= y[i] - np.sum([coefficient * self.apply_basis_fun(x[i], basis)
                                           for coefficient, basis in zip(coefficients, self.bases)])
            model += u[i] >= - (y[i] - np.sum([coefficient * self.apply_basis_fun(x[i], basis)
                                              for coefficient, basis in zip(coefficients, self.bases)]))
        model += lpSum(u)
        status = model.solve(getSolver('PULP_CBC_CMD', msg=False))
        # status = model.solve()
        assert status == 1, 'LAD solution not optimal'
        return [c.varValue for c in coefficients]

    def _get_ols_refined_coefficients(self, data):
        self.ols_points = data
        import statsmodels.api as sm
        try:
            X = sm.add_constant(np.stack([self.apply_basis_fun(data.roi, basis) for basis in self.bases], axis=1))
            y = data.rt
            model = sm.OLS(y, X)
            return model.fit().params
        except: # with too few data points: error
            warning('not enough data points for OLS model')

    def apply_basis_fun(self, x, basis):
        match basis:
            case '1':
                return np.ones_like(x)
            case 'x':
                return x
            case 'x**2':
                return x ** 2
            case 'sqrt(x)':
                return np.sqrt(x)
            case 'x*sqrt(x)':
                return x * np.sqrt(x)

    def get_mapping(self, x):
        result = np.zeros_like(x)
        for coefficient, basis in zip(self.coefficients, self.bases):
            result += coefficient * self.apply_basis_fun(x, basis)
        return result

if __name__ == '__main__':
    import pandas as pd
    ols_neg_data = pd.read_csv('/home/fleming/Documents/Projects/rtranknet/runs/FE_sys_eval/predictions/FE_setup_disjoint_sys_yes_cluster_yes_fold1_ep10_0271.tsv',
                               sep='\t', header=None, names=['smiles', 'rt', 'roi'])
    mini_data = pd.DataFrame({'roi': [1.], 'rt': [1.]})
    import doctest
    doctest.testmod(globs=globals() | dict(ols_neg_data=ols_neg_data, mini_data=mini_data))
