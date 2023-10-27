"""wrapper around dmpnn model"""
from chemprop.args import TrainArgs
from chemprop.models.mpn import MPN


def dmpnn(encoder_size, depth, dropout_rate):
    args = TrainArgs()
    args.from_dict({'dataset_type': 'classification',
                    'data_path': None,
                    'hidden_size': encoder_size,
                    'depth': depth,
                    'dropout': dropout_rate})
    model = MPN(args)
    model.name = 'dmpnn'
    return model
