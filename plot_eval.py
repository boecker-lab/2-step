import json
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import argparse
from re import sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', nargs='+')
    parser.add_argument('--basename', default=None)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()
    jsons = {}
    for j in args.jsons:
        base = os.path.basename(os.path.splitext(j)[0])
        if (args.basename is not None):
            base = sub(rf'^{args.basename}', '', base)
        base = sub(r'_eval$', '', base)
        jsons[base] = json.load(open(j))
    test_ids = list(set.intersection(*(set(jsons[j]) for j in jsons)))
    model_names = list(jsons)
    accs = pd.DataFrame({'name': model_names} | {test_id: [jsons[m][test_id]['acc'] for m in model_names]
                                                 for test_id in test_ids}).set_index('name')
    # print(accs)
    # print(accs.transpose())
    print(accs.transpose().agg(['mean', 'median']).transpose())
    accs.transpose().boxplot(figsize=(15, 5))
    if args.save is None:
        plt.show()
    else:
        plt.savefig(args.save, bbox_inches='tight')
