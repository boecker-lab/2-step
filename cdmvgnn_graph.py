import sys
sys.path.append('../CD-MVGNN')
from dglt.data.featurization.mol2graph import mol2graph

def cdmvgnn_graph(smiles):
    return mol2graph([smiles], {}, None)
