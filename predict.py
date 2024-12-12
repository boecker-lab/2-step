from logging import basicConfig, INFO, info, warning
import pandas as pd
import numpy as np
import pickle
import json
import re
from tap import Tap
from typing import List, Optional, Literal, Tuple, Union
import pickle
import io
import torch
import yaml

from utils import Data
from mapping import LADModel

class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes' and not torch.cuda.is_available():
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_model(path: str, all_in_one:bool=False):
    path = path + '.pt' if not path.endswith('pt') else path
    if (torch.cuda.is_available()):
        model = torch.load(path, weights_only=False)
    else:
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        if hasattr(model, 'ranknet_encoder'):
            model.ranknet_encoder.embedding.gnn.device = torch.device('cpu')
            model.ranknet_encoder.embedding.gnn.encoder[0].device = torch.device('cpu')
        try:
            model.encoder.encoder[0].device = torch.device('cpu')
        except:
            pass
    if (all_in_one):
        return model
    path = re.sub(r'_ep\d+(\.pt)?$', '', re.sub(r'\.pt$', '', path)) # for ep_save
    data = DataUnpickler(open(f'{path}_data.pkl', 'rb')).load()
    config = json.load(open(f'{path}_config.json'))
    return model, data, config

class PredictArgs(Tap):
    input_compounds: str        # TSV file with `smiles` and `rt` columns
    input_metadata: str         # yaml file with at least `column.name`, `eluent.A.pH`, and `column.t0` specified
    out: str                    # where to write the output (TSV format)
    model: str                  # model to load
    gpu: bool = False           # whether to use GPU for predictions
    batch_size: int = 256       # adjust according to available VRAM
    verbose: bool = False       # more info on what is being done internally

if __name__ == '__main__':
    args = PredictArgs().parse_args()
    # args = PredictArgs().parse_args('--input_compounds /home/fleming/Documents/Projects/rtranknet/test/input_0002.tsv --input_metadata /home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/processed_data/0002/0002_metadata.yaml --out /home/fleming/Documents/Projects/rtranknet/test/predicted_0002.tsv --model /home/fleming/Documents/Projects/rtranknet/test/FE_setup_disjoint_sys_yes_cluster_no_fold1_new.pt --verbose'.split())
    if (args.verbose):
        basicConfig(level=INFO)
    if (args.gpu):
        torch.set_default_device('cuda')

    # load model
    info('load model...')
    model = load_model(args.model, all_in_one=True)
    data_args = model.extra_storage['data_args']
    sysfeature_scaler = model.extra_storage['sysfeature_scaler']

    info('load input data...')
    d = Data(**data_args)
    metadata = yaml.load(open(args.input_metadata), yaml.SafeLoader)
    # flatten metadata
    [metadata] = pd.json_normalize(metadata, sep='.').to_dict(orient='records')
    original_input_columns = open(args.input_compounds).readlines()[0].strip().split('\t')
    d.add_external_data(args.input_compounds, metadata=metadata,
                        remove_nan_rts=False, tab_mode=True,
                        isomeric=True, split_type='evaluate')

    # TODO: warn about missing metadata (or even error?)

    info('computing features')
    d.compute_features(mode=None, add_descs=False)

    info('computing graphs')
    d.compute_graphs()
    info('(fake) splitting data')
    d.split_data((0, 0))
    if (sysfeature_scaler is not None):
        info('standardize data')
        d.standardize(other_descriptor_scaler=None, other_sysfeature_scaler=sysfeature_scaler,
                      can_create_new_scaler=False)
    ((train_graphs, train_x, train_sys, train_y),
     (val_graphs, val_x, val_sys, val_y),
     (test_graphs, test_x, test_sys, test_y)) = d.get_split_data()
    X = np.concatenate((train_x, test_x, val_x)).astype(np.float32)
    X_sys = np.concatenate((train_sys, test_sys, val_sys)).astype(np.float32)
    Y = np.concatenate((train_y, test_y, val_y))
    info(f'done preprocessing. predicting ROIs...')
    graphs = np.concatenate((train_graphs, test_graphs, val_graphs))
    if (hasattr(model, 'add_sys_features') and model.add_sys_features):
        from utils_newbg import sysfeature_graph
        info('add system features to graphs')
        smiles_list = d.df.iloc[np.concatenate((d.train_indices, d.test_indices, d.val_indices))]['smiles'].tolist()
        assert len(graphs) == len(smiles_list)
        from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
        if (model.add_sys_features_mode == 'bond'):
            set_extra_bond_fdim(train_sys.shape[1])
        elif (model.add_sys_features_mode == 'atom'):
            set_extra_atom_fdim(train_sys.shape[1])
        for i in range(len(graphs)):
            graphs[i] = sysfeature_graph(smiles_list[i], graphs[i], X_sys[i],
                                         bond_or_atom=model.add_sys_features_mode)
    preds = model.predict(graphs, X, X_sys, batch_size=args.batch_size,
                          ret_features=False, prog_bar=args.verbose)
    info(f'done predicting ROIs. predicting retention times...')
    d.df['roi'] = preds[np.arange(len(d.df.rt))[ # restore correct order
        np.argsort(np.concatenate([d.train_indices, d.test_indices, d.val_indices]))]]
    # predict retention times right here
    d.df['roi2'] = d.df.roi ** 2 # for LAD model
    # anchors are all data points with annotated retention time, discarding the void volume
    data_anchors = d.df.loc[d.df.rt > metadata['column.t0']]
    data_to_predict = d.df.loc[pd.isna(d.df.rt)].copy()
    info(f'building mapping using {len(data_anchors)} anchors, predicting {len(data_to_predict)} retention times...')
    mapping_model = LADModel(data_anchors, ols_after=True, ols_discard_if_negative=True, ols_drop_mode='2*median')
    data_to_predict['rt_pred'] = mapping_model.get_mapping(data_to_predict.roi)
    # TODO: output anchors, too?
    info(f'done. saving to {args.out}.')
    data_to_predict[
        # [c for c in data_to_predict.columns if any(['smiles' in c, 'inchi' in c.lower(), 'name' in c, c.startswith('rt_pred'), c.startswith('id')])]
        original_input_columns + ['rt_pred']
                    ].to_csv(args.out, sep='\t')
