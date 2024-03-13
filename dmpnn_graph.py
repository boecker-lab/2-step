from chemprop.features import MolGraph, BatchMolGraph

def dmpnn_graph(smiles, atom_features_extra=None, bond_features_extra=None):
    return MolGraph(smiles, atom_features_extra=atom_features_extra,
                    bond_features_extra=bond_features_extra)

def dmpnn_batch(graphs):
    return  BatchMolGraph(graphs)

if __name__ == '__main__':
    graphs = [dmpnn_graph(s) for s in ['CCCN', 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O']]
    batch = dmpnn_batch(graphs)
    import numpy as np
    graphs_sys = [dmpnn_graph(s, bond_features_extra=np.array([[0.2, 0.3, 1. , 0.5]] * n_bonds),)
                 for s, n_bonds in zip(['CCCN', 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O'],
                                       [3, 12])]
    from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
    set_extra_bond_fdim(4)
    set_extra_bond_fdim(0)
    batch_sys = dmpnn_batch(graphs_sys)
