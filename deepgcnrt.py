import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from rdkit import Chem
from deepgcnrt_features import get_node_dim, get_edge_dim

class GlobalPool(nn.Module):
    """One-step readout in AttentiveFP

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """Perform one-step readout

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a']
            else:
                return self.gru(context, g_feats)

class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class computes graph representations out of node features.

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        """Computes graph representations out of node features.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.

        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats

class EmbeddingLayerConcat(nn.Module):
    def __init__(self, node_in_dim, node_emb_dim, edge_in_dim=None, edge_emb_dim=None):
        super(EmbeddingLayerConcat, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_emb_dim= node_emb_dim
        self.edge_in_dim = edge_emb_dim
        self.edge_emb_dim=edge_emb_dim

        self.atom_encoder = nn.Linear(node_in_dim, node_emb_dim)
        if edge_emb_dim is not None:
            self.bond_encoder = nn.Linear(edge_in_dim, edge_emb_dim)

    def forward(self, g):
        node_feats, edge_feats= g.ndata["node_feat"], g.edata["edge_feat"]
        node_feats = self.atom_encoder(node_feats)

        if self.edge_emb_dim is None:
            return node_feats
        else:
            edge_feats = self.bond_encoder(edge_feats)
            return  node_feats, edge_feats

"adopted and modified from: https://lifesci.dgl.ai/_modules/dgllife/model/gnn/gcn.html#GCN"
class GCNLayerWithEdge(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None,
                 residual=True, output_norm="none", dropout=0., update_func="no_relu"):
        super(GCNLayerWithEdge, self).__init__()

        self.activation = activation
        self.mlp = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.aggr = update_func # relu, relu_eps_beta, no_relu

        if self.aggr == "relu_eps_beta":
            #for relu eps beta
            self.eps=1e-7
            self.beta = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        if output_norm == "batch_norm":
            self.bn_layer = nn.BatchNorm1d(out_feats)
            self.output_norm = True
        elif output_norm == "layer_norm":
            self.bn_layer = nn.LayerNorm(out_feats)
            self.output_norm = True
        elif output_norm == "none":
            self.output_norm = False
        else:
            raise NotImplementedError

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # Node and edge feature dimension need to match.
            g.ndata['h'] = node_feats
            g.edata['h'] = edge_feats
            g.apply_edges(fn.u_add_e('h', 'h', 'm'))


            if self.aggr == 'relu_eps_beta':
                g.edata['m'] = F.relu(g.edata['m']) + self.eps
                g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            elif self.aggr == "no_relu":
                g.edata['a'] = edge_softmax(g, g.edata['m'])
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            elif self.aggr == "relu":
                # relu activation; have softmax aggration
                g.edata['m'] = F.relu(g.edata['m'])
                g.edata['a'] = edge_softmax(g, g.edata['m'])
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            else:
                raise NotImplementedError

            new_feats = g.ndata['m']
            new_feats = self.mlp(new_feats)
            new_feats = self.activation(new_feats)
            new_feats = self.dropout(new_feats)

            if self.residual:
                new_feats = new_feats + node_feats
            if self.output_norm:
                new_feats = self.bn_layer(new_feats)

            return new_feats


'''GCN model with edge, attention and GRU readout'''
class GCNModelWithEdgeAFPreadout(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_feats=None, activation=F.relu,
                 residual=True, output_norm="none", dropout=0.1, gru_out_layer=2, update_func="no_relu"):
        super(GCNModelWithEdgeAFPreadout, self).__init__()

        if hidden_feats is None:
            hidden_feats = [200]*5

        in_feats = hidden_feats[0]
        n_layers = len(hidden_feats)

        activation = [activation for _ in range(n_layers)]
        residual = [residual for _ in range(n_layers)]
        output_norm = [output_norm for _ in range(n_layers)]
        dropout = [dropout for _ in range(n_layers)]

        lengths = [len(hidden_feats), len(activation),
                   len(residual), len(output_norm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.embed_layer = EmbeddingLayerConcat(node_in_dim, hidden_feats[0], edge_in_dim, hidden_feats[0])
        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayerWithEdge(in_feats, hidden_feats[i], activation[i], residual[i], output_norm[i], dropout[i], update_func))
            in_feats = hidden_feats[i]

        self.readout = AttentiveFPReadout(
            hidden_feats[-1], num_timesteps=gru_out_layer, dropout=dropout[-1]
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g):
        node_feat, edge_feat = self.embed_layer(g)
        for gnn in self.gnn_layers:
            node_feat = gnn(g, node_feat, edge_feat)
        # g.ndata['feats'] = feats
        # feats = dgl.sum_nodes(g, "feats")
        # feats = self.out(feats)
        out = self.readout(g, node_feat)
        return out

def deepgcnrt(num_layers=16, hid_dim=200, dropout=0.1):
    model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(),
                                      edge_in_dim=get_edge_dim(),
                                      hidden_feats=[hid_dim]*num_layers,
                                      output_norm='none',
                                      gru_out_layer=2,
                                      update_func='no_relu',
                                      dropout=dropout,
                                      residual=True)
    model.device = model.embed_layer.atom_encoder.bias.device
    model.name = 'deepgcnrt'
    return model

if __name__ == '__main__':
    from deepgcnrt_graph import deepgcnrt_graph

    smiles = ['C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
              'C([C@@H]1[C@H]([C@@H]([C@@H](C(O1)O)O)O)O)O',
              'CCCN',           # Propylamine
              ]
    graphs = [deepgcnrt_graph(s) for s in smiles]

    model = deepgcnrt()
    model(graphs[0])
