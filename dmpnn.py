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

if __name__ == '__main__':
    model = dmpnn(200, 3, 0)
    from dmpnn_graph import dmpnn_graph, dmpnn_batch
    graphs = [dmpnn_graph(s) for s in ['CCCN', 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O']]
    batch = dmpnn_batch(graphs)
    model([batch])
