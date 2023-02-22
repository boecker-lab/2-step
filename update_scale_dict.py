import sys
import json
from utils import REL_COLUMNS
from hashlib import md5
import os.path
from pprint import pprint

HSM_FIELDS = ['H', 'S*', 'A', 'B', 'C (pH 2.8)', 'C (pH 7.0)']
TANAKA_FIELDS = ['kPB', 'αCH2', 'αT/O', 'αC/P', 'αB/P', 'αB/P.1']

if __name__ == '__main__':
    repo_root_folder = sys.argv[1]
    sys.path.append(repo_root_folder)
    from pandas_dfs import get_dataset_df
    dss = get_dataset_df()
    scales = {}
    # 1. add known scaling: 'eluent.', 'gradient.'
    for field in dss.columns:
        if ((field.startswith('eluent.'))
            and '.unit' not in field):
            print(field, 'MANUAL')
            if '.pH' in field:
                scales[field] = {'mean': 7., 'std': 7.}
            else:
                scales[field] = {'mean': 50., 'std': 50.}
    # 2. add column params (REL_COLUMNS)
    for field in REL_COLUMNS:
        if field not in scales:
            m, s = dss[field].agg(['mean', 'std'])
            scales[field] = {'mean': m, 'std': s}
    # 3. add HSM fields
    for field in HSM_FIELDS:
        if field not in scales:
            m, s = dss[field].agg(['mean', 'std'])
            scales[field] = {'mean': m, 'std': s}
    # 3. add tanaka fields
    for field in TANAKA_FIELDS:
        if field not in scales:
            m, s = dss[field].agg(['mean', 'std'])
            scales[field] = {'mean': m, 'std': s}
    # dump
    pprint(scales)
    out = os.path.join(repo_root_folder, 'scaling.json')
    json.dump(scales, open(os.path.join(repo_root_folder, 'scaling.json'), 'w'))
    print('md5', md5(open(out, 'rb').read()).hexdigest())
