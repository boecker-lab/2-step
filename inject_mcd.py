"""with prediction TSVs and eval.jsons, compute Minimum Compound Deletion + ratio and put into the evals
"""

import argparse
import json
from os.path import basename, join, exists
import pandas as pd
from evaluate import lcs_results, NpEncoder

def get_void(ds, repo_root, void_factor=2):
    return pd.read_csv(join(repo_root, f'processed_data/{ds}/{ds}_metadata.tsv'), sep='\t')['column.t0'].iloc[0] * void_factor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_dir', required=True)
    parser.add_argument('--repo_root', required=True)
    parser.add_argument('eval_jsons', nargs='+')
    parser.add_argument('--warn_lcs_diff', action='store_true')
    args = parser.parse_args()
    for f in args.eval_jsons:
        assert f.endswith('_eval.json'), f'{f} no valid eval json'
        j = json.load(open(f))
        base = basename(f).replace('_eval.json', '')
        for ds in j:
            pred_paths = [join(args.roi_dir, base, f'{base}_{ds}.tsv'), join(args.roi_dir, f'{base}_{ds}.tsv')]
            if (exists(pred_paths[0])):
                pred_path = pred_paths[0]
            else:
                pred_path = pred_paths[1]
            df_pred = pd.read_csv(pred_path, sep='\t',
                                  names=['smiles', 'rt', 'roi'], header=None)
            df_mcd = df_pred.loc[df_pred.rt > get_void(ds, args.repo_root)]
            mcd = len(df_mcd) - lcs_results(df_mcd, 'lis')
            mcd_ratio = mcd / (len(df_mcd) - 1) # subtract one because a single compound cannot be in conflict
            j[ds]['mcd'] = mcd
            j[ds]['mcd_ratio'] = mcd_ratio
            if (args.warn_lcs_diff and 'lcs_dist' in j[ds] and int(j[ds]['lcs_dist']) != int(mcd)):
                print('WARNING', base, ds, f'{j[ds]["lcs_dist"]=}', '!=', f'{mcd=}')
        # write
        out_path = f.replace('_eval.json', '_eval_mcd.json')
        with open(out_path, 'w') as out:
            json.dump(j, out, indent=2, cls=NpEncoder)
        print(out_path)
