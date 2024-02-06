from typing import Iterable
from ogb.utils import smiles2graph
from torch_geometric.data import Data, Batch
from torch.utils.data import default_convert

# stored as normal numpy arrays
def gnn_graph(smiles):
    graph = smiles2graph(smiles)
    # return Data(**graph)
    return graph

# transformes numpy graphs to torch_geometric Data, then to Batch
def gnn_batch(graphs):
    data = []
    for g in graphs:
        data.append(Data(x=default_convert(g['node_feat']),
                         edge_index=default_convert(g['edge_index']),
                         edge_attr=default_convert(g['edge_feat'])))
    batch = Batch.from_data_list(data)
    # batch['edge_index'] = default_convert(batch['edge_index'])
    # batch['edge_feat'] = default_convert(batch['edge_feat'])
    # batch['node_feat'] = default_convert(batch['node_feat'])
    return batch

if __name__ == '__main__':
    smiles_list = [
        'CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3',
        'CC(C)(C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O)C(=O)O',
        'CC12CCC3C(C1CCC2C(=O)NC(C)(C)C)CCC4C3(C=CC(=O)N4)C',
        'C1CCNC(C1)CNC(=O)C2=C(C=CC(=C2)OCC(F)(F)F)OCC(F)(F)F',
        'C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O',
        'CC1CC2C3CC(C4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C)F',
        'C1CN(CCN1CC=CC2=CC=CC=C2)C(C3=CC=C(C=C3)F)C4=CC=C(C=C4)F',
        'CN1C(=O)CN=C(C2=C1C=CC(=C2)[N+](=O)[O-])C3=CC=CC=C3F',
        'CC1=C(C=CC=C1NC2=C(C=CC=N2)C(=O)O)C(F)(F)F',
        'CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F',
        'CC12CCC(=O)C=C1CCC3C2(C(CC4(C3CCC4(C)O)C)O)F',
        'C1CN(CCN1CCC=C2C3=CC=CC=C3SC4=C2C=C(C=C4)C(F)(F)F)CCO',
        'C1CN(CCN1CCCN2C3=CC=CC=C3SC4=C2C=C(C=C4)C(F)(F)F)CCO',
        'CCN(CC)CCN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3F',
        'CC(C1=CC(=C(C=C1)C2=CC=CC=C2)F)C(=O)O',
        'COCCCCC(=NOCCN)C1=CC=C(C=C1)C(F)(F)F',
        'C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl',]
    graphs = list(map(gnn_graph, smiles_list))
    print(f'{graphs[0]=}')
    batch = gnn_batch(graphs)
    assert all((batch[i].x.numpy() == graphs[i]['node_feat']).all() and
               (batch[i].edge_index.numpy() == graphs[i]['edge_index']).all() and
               (batch[i].edge_attr.numpy() == graphs[i]['edge_feat']).all()
               for i in range(len(graphs)))
