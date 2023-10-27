import sys
import os.path
dir_ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ + '/CD-MVGNN')
from dglt.models.zoo.mpnn import DualMPNNPlus, DualMPNN
from argparse import Namespace

def get_cdmvgnn_args(encoder_size=300, depth=3, dropout_rate=0.0):
    args = Namespace()
    args.depth = depth
    args.dropout = dropout_rate
    args.output_size = encoder_size
    # default args
    args.atom_messages = False
    args.features_only = False
    args.no_attach_fea = False
    args.bias = False
    args.undirected = False
    args.dense = False
    args.dataset_type = 'regression'
    args.hidden_size = 2
    args.self_attention = True      # default False
    args.attn_out = 128
    args.attn_hidden = 4
    args.cuda = False
    args.use_input_features = False
    args.activation = 'ReLU'
    args.ffn_num_layers = 1
    args.ffn_hidden_size = 1
    args.dist_coff = 0.1
    args.coord = None               # something 3d coord
    args.no_cache = False
    args.input_layer = 'fc'
    args.bond_drop_rate = 0
    args.atom_messages = False
    args.use_norm = True            # for DualMPNN without Plus
    return args


def cdmvgnn(encoder='DualMPNNPlus', encoder_size=300, depth=3, dropout_rate=0.0,
            args=None):
    if (args is None):
        args = get_cdmvgnn_args(encoder_size=encoder_size, depth=depth, dropout_rate=dropout_rate)
    if (encoder.lower() == 'dualmpnnplus'):
        model = DualMPNNPlus(args)
    elif (encoder.lower() == 'dualmpnn'):
        model =  DualMPNN(args)
    else:
        raise NotImplementedError(encoder)
    # TODO: for now
    model.device = model.encoders.encoder.W_node.weight.device
    model.name = encoder
    return model
