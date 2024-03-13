"""wrapper around dmpnn model"""
from chemprop.args import TrainArgs
from chemprop.models.mpn import MPN
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim


def dmpnn(encoder_size, depth, dropout_rate, add_sys_features=False,
          add_sys_features_mode=None, add_sys_features_dim=None):
    args = TrainArgs()
    args.from_dict({'dataset_type': 'classification',
                    'data_path': None,
                    'hidden_size': encoder_size,
                    'depth': depth,
                    'dropout': dropout_rate})
    if (add_sys_features):
        if (add_sys_features_mode == 'bond'):
            set_extra_bond_fdim(add_sys_features_dim)
        elif (add_sys_features_mode == 'atom'):
            set_extra_atom_fdim(add_sys_features_dim)
        else:
            raise NotImplementedError(f'{add_sys_features_mode=}')
    model = MPN(args)
    model.name = 'dmpnn'
    return model

if __name__ == '__main__':
    model = dmpnn(200, 3, 0)
    from dmpnn_graph import dmpnn_graph, dmpnn_batch
    graphs = [dmpnn_graph(s) for s in ['CCCN', 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O']]
    batch = dmpnn_batch(graphs)
    model([batch])
    # with or without added sys features
    m_sans = dmpnn(300, 3, 0.1)
    m_sys = dmpnn(300, 3, 0.1, add_sys_features=True, add_sys_features_mode='atom', add_sys_features_dim=12)
