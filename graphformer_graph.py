# repurpose deepgcnrt_graph.py
# goal:
"""
{'edge_index': [[2 * n_edges], [2 * n_edges]],
'edge_attr': [2 * n_edges x edge_dim],
'num_nodes',
'node_feat': [n_nodes x node_dim],
}
"""

"""
atom_features = [
    'chiral_center',
    'cip_code',
    'crippen_log_p_contrib',
    'crippen_molar_refractivity_contrib',
    'degree',
    'element',
    'formal_charge',
    # 'gasteiger_charge',
    'hybridization',
    'is_aromatic',
    'is_h_acceptor',
    'is_h_donor',
    'is_hetero',
    'is_in_ring_size_n',
    'labute_asa_contrib',
    'mass',
    'num_hs',
    'num_radical_electrons',
    'num_valence',
    'tpsa_contrib',
]

bond_features = [
    'bondstereo',
    'bondtype',
    'is_conjugated',
    'is_in_ring',
    'is_rotatable',
]
TODO: atom: is_in_ring, chirality, element_number
"""

from deepgcnrt_graph import smiles2graph
from deepgcnrt_features import atom_features, bond_features
from transformers.models.graphormer.collating_graphormer import preprocess_item

def graphformer_graph(smiles_string):
    graph = smiles2graph(
        smiles_string, exclude_node=[f for f in atom_features if f not in [
            'atom_num', 'chiral_tag', 'formal_charge', 'is_in_ring_atom']],
        exclude_edge=[f for f in bond_features if f not in [
            'bondtype_num', 'bondstereo_num', 'is_conjugated']])
    graph['labels'] = [0.]
    return preprocess_item(graph)
