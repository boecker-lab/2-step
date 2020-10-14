import obonet
import networkx
import numpy as np

graph = obonet.read_obo('ChemOnt_2_1.obo')
encodings = {}
for node in graph.nodes:
    rank = len(networkx.descendants(graph, node)) - 1
    encodings.setdefault(rank, {})[node] = len(
        encodings[rank]) if rank in encodings else 0


id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {data['name']: id_ for id_,
              data in graph.nodes(data=True) if 'name' in data}


def get_onehot(oid, rank=None):
    global encodings
    if rank is None:
        for r in encodings:
            if oid in encodings[r]:
                rank = r
        if (rank is None):
            raise Exception('rank not found!')
    return (np.eye(len(encodings[rank]))[encodings[rank][oid]]
            if (oid in encodings[rank])
            else np.zeros((len(encodings[rank]), )))

def get_binary(oid, l_thr=0.005, u_thr=0.25):
    if not hasattr(get_binary, 'occs'):
        get_binary.occs = [(l.split('\t')[0], float(l.strip().split('\t')[1])) for l in open('chemont_occs.tsv')]
        get_binary.classes = [x[0] for x in get_binary.occs if x[1] >= l_thr and x[1] <= u_thr]
        get_binary.classes_m = np.eye(len(get_binary.classes))
        get_binary.zeros = np.zeros(len(get_binary.classes))
    if (isinstance(oid, str)):
        return (get_binary.classes_m[get_binary.classes.index(oid)] if oid in get_binary.classes
                else get_binary.zeros)
    return np.array([c in oid for c in get_binary.classes]).astype(np.float)
    
