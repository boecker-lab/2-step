"""with prediction TSVs and eval.jsons, recompute accuracy and put into json
"""

import argparse
import json
from os.path import basename, join, exists
import pandas as pd
import numpy as np
from evaluate import eval_, NpEncoder

def get_void(ds, repo_root, void_factor=2):
    return pd.read_csv(join(repo_root, f'processed_data/{ds}/{ds}_metadata.tsv'), sep='\t')['column.t0'].iloc[0] * void_factor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_dir', required=True)
    parser.add_argument('--repo_root', required=True)
    parser.add_argument('--epsilon', type=float, default=10/60)
    # parser.add_argument('--dont_count_low_epsilon', action='store_true')
    parser.add_argument('--print_acc_diffs', action='store_true')
    parser.add_argument('eval_jsons', nargs='+')
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
            void_rt = get_void(ds, args.repo_root)
            df_novoid = df_pred.loc[df_pred.rt > void_rt]
            acc = eval_(df_pred.rt, df_pred.roi, args.epsilon, void_rt=void_rt, dont_count_low_epsilon=True)
            j[ds]['acc_ignore_epsilon'] = acc
            if (args.print_acc_diffs):
                print(base, ds, f'accuracy difference: {np.abs(j[ds]["acc_ignore_epsilon"] - j[ds]["acc"]):.1%}')
        # write
        out_path = f.replace('_eval.json', '_eval_acc_ignore_epsilon.json')
        with open(out_path, 'w') as out:
            json.dump(j, out, indent=2, cls=NpEncoder)
        print(out_path)
