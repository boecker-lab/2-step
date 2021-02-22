from features import features
from utils import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_features():
    smiles = ['C1CC(=O)N[C@@H]1C(=O)O',
              'CC(=O)O[C@H](CC(=O)[O-])C[N+](C)(C)C',
              'C1C[C@H](N(C1)C(=O)C[NH3+])C(=O)[O-]'
              'C1=CC=C(C=C1)C[C@@H](C(=O)O)N', 'CC(C)C[C@@H](C(=O)O)NC(=O)C',
              'C1CC(=O)N[C@@H]1C(=O)O', 'CSCC[C@@H](C(=O)O)N',
              'C/C=C(\C)/C(=O)NCC(=O)O', 'C[N+]1(CCC[C@H]1C(=O)[O-])C',
              'C1=NC2=C(N1[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)NC(=O)NC2=O',
              'CC(=O)NC1=NC(=O)N(C=C1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO)O)O',
              'C1=CC=C2C(=C1)C(=CN2)C(=O)O[C@H]3[C@@H]([C@H]([C@@H](C(O3)C(=O)O)O)O)O',
              'CNC1=NC2=C(C(=O)N1)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O',
              'C1=CC=C(C=C1)[C@@H]([C@@H](C(=O)[O-])[NH3+])O',
              'C1=CC(=CC=C1C[C@@H](C(=O)O)N)O',
              'C1=CC=C(C=C1)[C@@H]([C@@H](C(=O)[O-])[NH3+])O',
              'CN1[C@@H](C[C@H](C1=O)O)C2=CN=CC=C2', 'CN1[C@@H](CCC1=O)C2=CN=CC=C2',
              'CC(=O)N[C@@H](CCCC(=O)O)C(=O)O', 'C1=CC=C(C=C1)/C=C/C(=O)NCC(=O)O',
              'CC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O',
              'C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N[C@@H](CC(=O)O)C(=O)O',
              'C[C@]12CC[C@H](C[C@@H]1CC[C@@H]3[C@@H]2CC[C@]4([C@H]3CCC4=O)C)O[C@H]5[C@@H]([C@H]([C@@H]([C@H](O5)C(=O)O)O)O)O',
              'C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N',
              'CC(=O)NCCC(=O)N[C@@H](CC1=CN=CN1)C(=O)O',
              'CCCCC/C=C/C(=O)OC(CC(=O)[O-])C[N+](C)(C)C',
              'C1[C@@H]([C@H]([C@@H]([C@H](N1)CO)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)O)O)O)O',
              'C1[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)OP(=O)(O1)O',
              'CC1=CC2=C(C=C1C)N(C3=C(N2)C(=O)NC(=O)N3)C[C@@H]([C@@H]([C@@H](COP(=O)(O)O)O)O)O',
              'C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)C(=O)O)O)O)O)CCC5=CC(=O)CC[C@]35C',
              'C[C@@H](C(=O)O)NC(=O)C',]
    smiles = list(set(smiles))
    features.cached = {}
    res = features(smiles, custom_features=['MolLogP'])
    mols = []
    for s in smiles:
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(s))
            AllChem.EmbedMolecule(mol)
        except:
            mol = None
        mols.append(mol)
    res_man = {(s, 'MolLogP'): Chem.Crippen.MolLogP(mol)
               if mol is not None else np.nan
               for s, mol in zip(smiles, mols)}
    assert all(((r1:=features.cached[(s, 'MolLogP')]) == (r2:=res_man[(s, 'MolLogP')]))
               or (np.isnan(r1) and np.isnan(r2))
               for s in smiles), 'pooled calculation is faulty!'
    # comm
    res2 = features(smiles, 'rdk')
    res2_old = features_old(smiles, 'rdk')
    print(res2)
    print(res2_old)
    return np.isclose(res2, res2_old, equal_nan=True).all()


def test_data_csv():
    csvfile = '/home/fleming/Documents/Projects/rtranknet/data/metlin_retention_times.csv'
    data = Data.from_raw_file(csvfile)

def test_void_est():
    from glob import glob
    from utils import naive_void_est
    rtdata = glob('/home/fleming/Documents/Projects/RtPredTrainingData/processed_data/*/*_rtdata_canonical_success.txt')
    nrows = len(rtdata) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 22))
    for f, ax in zip(rtdata, axes.ravel()):
        df = pd.read_csv(f, sep='\t', index_col=0)
        index = np.arange(0, len(df))
        void_i = [rt for rt in sorted(df.rt) if rt >= naive_void_est(df)][0]
        ax.plot(index, sorted(df.rt))
        ax.axvline(void_i)
    plt.savefig('void.pdf')
