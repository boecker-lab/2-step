import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
import multiprocessing as mp
import logging

logger = logging.getLogger('twosteprt.features')
info = logger.info
warning = logger.warning

def compute_descriptors(smile, descriptors):
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        AllChem.EmbedMolecule(mol)
    except:
        return [descriptors, [np.nan for _ in descriptors], descriptors]
    values = []
    failed = []
    for name in descriptors:
        fun = features.descriptors[name]
        try:
            val = fun(mol)
        except:
            val = np.nan
            failed.append(name)
        values.append(val)
    return [descriptors, values, failed]

def get_descriptors():
    features = []
    features.extend([(name, fun, 'rdk') for name, fun in Descriptors.descList])
    features.extend([(name, fun, '3d') for name, fun in
                     [('Asphericity', Descriptors3D.Asphericity),
                      ('Eccentricity', Descriptors3D.Eccentricity),
                      ('InertialShapeFactor', Descriptors3D.InertialShapeFactor),
                      ('NPR1', Descriptors3D.NPR1), ('NPR2', Descriptors3D.NPR2),
                      ('PMI1', Descriptors3D.PMI1), ('PMI2', Descriptors3D.PMI2), ('PMI3', Descriptors3D.PMI3),
                      ('RadiusOfGyration', Descriptors3D.RadiusOfGyration),
                      ('SpherocityIndex', Descriptors3D.SpherocityIndex)]])
    return features

def compute_morgan(smile, bits=1024, radius=2):
    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
    AllChem.EmbedMolecule(mol)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits))

def features(smiles, filter_='rdk', overwrite_cache=False, verbose=False,
             custom_features=[], mode='rdkit', load_factor=0.75,
             add_descs=False, add_desc_file='data/qm_merged.csv'):
    """computes and returns features as well as an ordered list of descriptors.

    If `custom_features` ~= ["morgan\d"], use morgan fingerprints, ignoring `filter_`
    """
    assert (len(smiles) == len(set(smiles))), 'smiles have to be unique'
    if (not hasattr(features, 'cached')):
        features.cached = {}
    out_arrays = []
    out_names = []
    if (mode is None):
        mode = ''
    if (mode.startswith('morgan')):
        bits = 1024
        radius = 2
        out_arrays.append(np.concatenate([compute_morgan(smile, bits, radius) for smile in smiles]))
        out_names.extend([f'morgan{i}' for i in range(bits)])
    if (mode.startswith('ae')):
        s1, s2 = list(map(int, mode[2:]))
        out = np.array([get_ae_features(smile, s1, s2) for smile in smiles])
        out_arrays.append(out)
        out_names.extend([f'ae_lstm_{s1}_{s2}_{i}' for i in range(out.shape[1])])
        # return (out,
        #         [f'ae_lstm_{s1}_{s2}_{i}' for i in range(out.shape[1])])
    if (mode == 'rdkit'):
        descriptors = get_descriptors()
        if (filter_ is not None):
            filter_fun = {'rdk': lambda t: t[2] == 'rdk',
                          '3d': lambda t: t[2] == '3d',
                          }[filter_]
            descriptors = list(filter(filter_fun, descriptors))
        if (len(custom_features) > 0):
            descriptors = sorted(list(filter(lambda t: t[0] in custom_features, descriptors)),
                              key=lambda t: custom_features.index(t[0]))
        features.descriptors = {name: fun for name, fun, _ in descriptors}
        to_calc = {}
        for s in smiles:
            for fname, ffun, _ in descriptors:
                if ((s, fname) not in features.cached
                    or overwrite_cache):
                    to_calc.setdefault(s, []).append(fname)
        to_calc = [(smile, descriptors) for smile, descriptors in to_calc.items()]
        if (len(to_calc) > 0):
            if (hasattr(features, 'write_cache')):
                features.write_cache = True # cache has to be written in the end
            nthreads = np.floor(mp.cpu_count() * load_factor).astype(int)
            if (verbose):
                info(f'calculating {len(to_calc)} feature values using {nthreads} threads')
            pool = mp.Pool(nthreads)
            res = pool.starmap(compute_descriptors, to_calc)
            pool.close()
            res_new = []
            for descs, values, failed in res:
                if (len(failed) > 0 and verbose):
                    info('failed', failed)
                res_new.append([(d, v) for d, v in zip(descs, values)])
            features.cached.update({(smile[0], desc): value for smile, smile_res in zip(to_calc, res_new)
                                    for desc, value in smile_res})
        out_arrays.append(np.array([[features.cached[(smile, desc[0])] for desc in descriptors]
                                    for smile in smiles]))
        out_names.extend([desc[0] for desc in descriptors])
        # return np.array([[features.cached[(smile, desc[0])] for desc in descriptors]
        #                  for smile in smiles]), [desc[0] for desc in descriptors]
    if (add_descs):
        out, names = get_add_descs(smiles, add_desc_file=add_desc_file)
        out_arrays.append(out)
        out_names.extend(names)
    if (len(out_arrays) > 0):
        out = np.concatenate(out_arrays, axis=1)
    else:
        out = np.array([]).reshape((len(smiles), 0))
    assert (out.shape[1] == len(out_names)), '#descriptor names ≠ #descriptors'
    return out, out_names



def parse_feature_spec(spec):
    if spec.startswith('rdk'):
        filter_ = None if spec == 'rdkall' else spec[3:]
        return {'mode': 'rdkit', 'filter_features': filter_}
    if (spec.lower() == 'none'):
        spec = None
    return {'mode': spec, 'filter_features': None}
