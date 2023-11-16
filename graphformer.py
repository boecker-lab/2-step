from transformers import GraphormerForGraphClassification
from transformers.models.graphormer.configuration_graphormer import GraphormerConfig
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

def graphformer(num_layers=12, hid_dim=768, dropout=0.1):
    config = GraphormerConfig()
    # TODO: settings
    # config.num_atoms = 440000
    config.num_hidden_layers = num_layers
    config.embedding_dim = hid_dim
    # config.ffn_embedding_dim = hid_dim
    config.hidden_size = hid_dim
    config.dropout = dropout
    model = GraphormerForGraphClassification(config)
    model.encoder.name = 'graphformer'
    return model.encoder


if __name__ == '__main__':
    # config = GraphormerConfig()
    # # config.num_atoms = 440000
    # model = GraphormerForGraphClassification(config)
    model = graphformer()
    from graphformer_graph import graphformer_graph
    g = graphformer_graph('CCCN')
    g = graphformer_graph('C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O')
    # TODO: evtl. als int
    g_prep = preprocess_item(g)
    c = GraphormerDataCollator()
    x = c([g_prep])
    model(**x).last_hidden_state.shape
    # get 34 graphs with batch size 32, try out how the batching works
    import pandas as pd
    graphs = [graphformer_graph(s) for s in pd.read_csv('/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/processed_data/0045/0045_rtdata_isomeric_success.tsv', sep='\t')['smiles.std'].tolist()]
    batches = c([preprocess_item(g) for g in graphs])
