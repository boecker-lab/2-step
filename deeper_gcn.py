import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv, aggr
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import unbatch

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = AtomEncoder(hidden_channels)
        self.edge_encoder = BondEncoder(hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)


        # self.graph_out = aggr.MeanAggregation()
        # self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, batch):
        x = self.node_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        edge_index = batch.edge_index

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return torch.stack([b.mean(0) for b in unbatch(x, batch.batch)])

def deeper_gcn(hid_dim=128, num_layers=8):
    m = DeeperGCN(hidden_channels=hid_dim, num_layers=num_layers)
    m.device = m.node_encoder.atom_embedding_list[0].weight.device
    m.name = 'deepergcn'
    return m

if __name__ == '__main__':
    m = deeper_gcn()
    # TODO: perhaps profile performance of mean/stacking here
    out = m(batch)
