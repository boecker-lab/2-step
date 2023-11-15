from chemprop.features import MolGraph, BatchMolGraph

def dmpnn_graph(smiles):
    return MolGraph(smiles)

def dmpnn_batch(graphs):
    return  BatchMolGraph(graphs)
