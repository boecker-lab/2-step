from chemprop.features import mol2graph

def dmpnn_graph(smiles):
    return mol2graph([smiles])
