"""takes pytorch model + data.pkl + config.json and puts everything into one file, to be used for `predict.py`"""
import sys
import torch

from predict import load_model

def get_data_args(data):
    data_args = {'use_compound_classes': data.use_compound_classes,
                 'use_system_information': data.use_system_information,
                 'metadata_void_rt': False,
                 'remove_void_compounds': False,
                 'classes_l_thr': data.classes_l_thr,
                 'classes_u_thr': data.classes_u_thr,
                 'use_usp_codes': data.use_usp_codes,
                 'custom_features': data.descriptors,
                 'use_hsm': data.use_hsm,
                 'use_ph': data.use_ph,
                 'use_gradient': data.use_gradient,
                 'use_newonehot': data.use_newonehot,
                 'custom_column_fields': data.custom_column_fields,
                 'columns_remove_na': False,
                 'hsm_fields': data.hsm_fields,
                 'graph_mode': True,
                 'encoder': 'dmpnn',
                 'remove_doublets': True}
    if (hasattr(data, 'use_tanaka')):
        data_args['use_tanaka'] = data.use_tanaka
    if (hasattr(data, 'tanaka_fields')):
        data_args['tanaka_fields'] = data.tanaka_fields
    if (hasattr(data, 'sys_scales')):
        data_args['sys_scales'] = data.sys_scales
    if (hasattr(data, 'solvent_order')):
        data_args['solvent_order'] = data.solvent_order
    return data_args


if __name__ == '__main__':
    in_path, out_path = sys.argv[1:]
    model, data, config = load_model(in_path)
    data_args = get_data_args(data)
    model.extra_storage = {'config': config, 'data_args': data_args,
                           'sysfeature_scaler': data.sysfeature_scaler}
    torch.save(model, out_path)
