import torch
import dgl
from rdkit import Chem
import numpy as np
from deepgcnrt_features import get_node_features, get_edge_dim, bond_featurizer

def smiles2graph(smiles_string,  exclude_node=None, exclude_edge=None):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    x = get_node_features(mol, exclude_node)
    # bond
    num_bond_features = get_edge_dim(exclude_edge)  #edge_dim
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_featurizer(bond, exclude_edge)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int32).T
        edge_attr = np.array(edge_features_list, dtype= np.float32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.float32)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def feature_to_dgl_graph(graph):
    '''
    input
    -------
    graph_dict, example:
    {'edge_feat': array([[0, 0, 0],
        [0, 0, 0]], dtype=int64), 'edge_index': array([[0, 1],
        [1, 0]], dtype=int64), 'node_feat': array([[5, 0, 4, 5, 3, 0, 2, 0, 0],
        [7, 0, 2, 5, 1, 0, 2, 0, 0]], dtype=int64), 'num_nodes': 2}
    output
    ---------
    #dgl_graph(int 32)
    '''
    g = dgl.graph((graph["edge_index"][0, :], graph["edge_index"][1, :]), num_nodes=graph["num_nodes"], idtype=torch.int32)
    g.ndata['node_feat'] = torch.tensor(graph['node_feat'], dtype=torch.float32)
    g.edata["edge_feat"] = torch.tensor(graph['edge_feat'], dtype=torch.float32)

    return g

def deepgcnrt_graph(smiles):
    return feature_to_dgl_graph(smiles2graph(smiles))
