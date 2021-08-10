import json
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import argparse
from re import sub, match

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', nargs='+')
    parser.add_argument('--basename', default=None)
    parser.add_argument('--save', default=None)
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()
    jsons = {}
    for j in args.jsons:
        base = os.path.basename(os.path.splitext(j)[0])
        if (args.basename is not None):
            base = sub(rf'^{args.basename}', '', base)
        base = sub(r'_eval$', '', base)
        jsons[base] = json.load(open(j))
    test_ids = list(set.intersection(*(set(jsons[j]) for j in jsons)))
    model_names = list(sorted(jsons, key=lambda x: int(sub(f'.*ep(\d+).*', r'\1', x)) if match(r'ep\d+', x) else 999))
    accs = pd.DataFrame({'name': model_names} | {test_id: [jsons[m][test_id]['acc'] for m in model_names]
                                                 for test_id in test_ids}).set_index('name')
    # print(accs)
    # print(accs.transpose())
    accs_simple = accs.transpose().agg(['mean', 'median']).transpose()
    print(accs_simple)
    # import pdb; pdb.set_trace()
    if (not args.no_plot):
        accs_simple.plot()
        plt.show()
        accs.transpose().boxplot(figsize=(15, 5))
        if args.save is None:
            plt.show()
        else:
            plt.savefig(args.save, bbox_inches='tight')
