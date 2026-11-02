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
import sys
from time import time

from utils import Data
from mapping import LADModel
from predict_rois import predict_rois

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
    input_compounds: str                 # TSV file with `smiles` and `rt` columns
    input_metadata: str                  # yaml file with at least `column.name`, `eluent.A.pH`, and `column.t0` specified
    model: str                           # model to load
    gpu: bool = False                    # whether to use GPU for predictions
    output_roi: bool = False             # include Retention Order Index (ROI) in output
    output_anchors: bool = False         # include anchors in output
    out: Optional[str] = None            # where to write the output (TSV format). If not specified, output will be written to screen.
    batch_size: int = 256                # adjust according to available VRAM
    repo_root_folder: str = '../RepoRT/' # location of RepoRT, needed for HSM/Tanaka database
    verbose: bool = False                # more info on what is being done internally

def metadata_incomplete(metadata):
    return (    # column not specified and neither HSM/Tanaka parameters
            ('column.name' not in metadata and
             any(f'column.{field}' not in metadata for field in Data().hsm_fields + Data().tanaka_fields))
            # or pH not specified
            or not any(ph_column in metadata for ph_column in
                       ['eluent.A.pH', 'eluent.B.pH']))

def load_metadata(path: str):
    metadata = yaml.load(open(path), yaml.SafeLoader)
    # flatten metadata
    [metadata] = pd.json_normalize(metadata, sep='.').to_dict(orient='records')
    if (metadata_incomplete(metadata)):
       warning(f'Important metadata is missing. Required: 1. Column name or manual HSM/Tanaka parameters, 2. pH. '
               f'Provided: {", ".join(metadata)}. '
               'This metadata is necessary; only the `setupagnostic` and `nocolumn` model will work. '
               'For these models the predictions will necessarily be worse!')
    return metadata

if __name__ == '__main__':
    args = PredictArgs().parse_args()
    if (args.verbose):
        basicConfig(level=INFO)
    if (args.gpu):
        torch.set_default_device('cuda')

    # load model
    info('load model...')
    model = load_model(args.model, all_in_one=True)

    info('load input data...')
    metadata = load_metadata(args.input_metadata)
    original_input_columns = open(args.input_compounds).readlines()[0].strip().split('\t')
    df = predict_rois(model, args.input_compounds, metadata, args.repo_root_folder,
                      args.batch_size, args.verbose)
    # anchors are all data points with annotated retention time, discarding the void volume
    data_anchors = df.loc[df.rt > metadata['column.t0']]
    data_to_predict = df.loc[pd.isna(df.rt)].copy()
    info(f'building mapping using {len(data_anchors)} anchors, predicting {len(data_to_predict)} retention times...')
    t0 = time()
    mapping_model = LADModel(data_anchors, ols_after=True, ols_discard_if_negative=True, ols_drop_mode='2*median')
    import pickle
    with open(args.input_compounds.replace('.tsv', '_model.pkl'), 'wb') as out:
        pickle.dump(mapping_model, out)
    data_to_predict['rt_pred'] = mapping_model.get_mapping(data_to_predict.roi)
    t1 = time() - t0
    info(f'took {t1:.1}s ({len(data_to_predict)/t1:.2}s per instance)')
    if (args.output_roi):
        original_input_columns += ['roi']
    out_df = data_to_predict[
        # [c for c in data_to_predict.columns if any(['smiles' in c, 'inchi' in c.lower(), 'name' in c, c.startswith('rt_pred'), c.startswith('id')])]
        original_input_columns + ['rt_pred']
    ]
    if (args.output_anchors):
        out_df = pd.concat([data_anchors[original_input_columns], out_df])
    if (args.out is None):
        info(f'showing output.')
        out_df.to_csv(sys.stdout, sep='\t')
    else:
        info(f'saving to {args.out}.')
        out_df.to_csv(args.out, sep='\t')
