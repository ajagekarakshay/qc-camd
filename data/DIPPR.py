import pandas as pd
import os.path as osp
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tensorflow.keras.utils import get_file
from tqdm import tqdm
from rdkit import Chem

from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot
from spektral.utils.io import load_csv, load_sdf

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 24, 32, 33, 35, 53, 80, 82]
BOND_TYPES = [1, 2, 3]

def mols_to_sdf():
    data = pd.read_csv("data\\DIPPR\\dippr_fp.csv")
    data = data.set_index("SMILES")
    writer = Chem.SDWriter('data\\DIPPR\\dippr_fp.sdf')
    for smi in data.index:
        mol = Chem.MolFromSmiles(smi)
        try:
            if mol.GetNumAtoms() > 1:
                writer.write(mol)
            else:
                data = data.drop([smi])
        except:
            data = data.drop([smi])
    data.to_csv("data\\DIPPR\\dippr_labels.csv")

# No non-zero charge or ISO

class DIPPR(Dataset): #Dataset
    def __init__(self, path, amount=None, n_jobs=1, **kwargs):
        self.path2 = path
        self.amount = amount
        self.n_jobs = n_jobs
        super().__init__(**kwargs)

    def read(self):
        print("Loading DIPPR dataset...")
        sdf_file = osp.join(self.path2, "dippr_fp.sdf")
        data = load_sdf(sdf_file, amount=self.amount)
        
        def read_mol(mol):
            x = np.array([atom_to_feature(atom) for atom in mol["atoms"]])
            a, e = mol_to_adj(mol)
            return x, a, e

        x_list, a_list, e_list = [], [], []
        # data = Parallel(n_jobs=self.n_jobs)(
        #     delayed(read_mol)(mol) for mol in tqdm(data, ncols=80)
        # )
        for mol in tqdm(data, ncols=80):
            try:
                x,a,e = read_mol(mol)
                x_list.append(x)
                a_list.append(a)
                e_list.append(e)
            except ValueError:
                pass

        #x_list, a_list, e_list = list(zip(*data))

        # Load labels
        labels_file = osp.join(self.path2, "dippr_labels.csv")
        labels = pd.read_csv(labels_file)
        labels = labels.set_index("SMILES").values
        if self.amount is not None:
            labels = labels[: self.amount]

        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]

def atom_to_feature(atom):
    atomic_num = label_to_one_hot(atom["atomic_num"], ATOM_TYPES)
    #coords = atom["coords"]
    #charge = atom["charge"]
    #iso = atom["iso"]
    return atomic_num

def mol_to_adj(mol):
    row, col, edge_attr, edges = [], [], {}, []
    for bond in mol["bonds"]:
        start, end = bond["start_atom"], bond["end_atom"]
        row += [start, end]
        col += [end, start]
        edge_attr[start, end] = bond["type"]
        edge_attr[end, start] = bond["type"]
    
    #print(row, col, edge_attr)
    a = sp.csr_matrix((np.ones_like(row), (row, col)))
    for row in range(a.shape[0]):
        for col in a[row].indices:
            #print(row, col)
            edges += [edge_attr[row,col]]

    #edge_attr = np.array([label_to_one_hot(e, BOND_TYPES) for e in edge_attr])
    edges = np.array([label_to_one_hot(e, BOND_TYPES) for e in edges])
    return a, edges

# dataset = DIPPR("data\\DIPPR\\dippr_fp.sdf")
# data = dataset.read()

# check_iso = False
# check_charge = False
# atom_types = set()
# bond_types = set()
# for mol in tqdm(data):
#     for atom in mol["atoms"]:
#         atom_types.add( atom["atomic_num"] )
#         if atom["iso"] != 0:
#             check_iso = True
#         if atom["charge"] != 0:
#             check_charge = True

#     for bond in mol["bonds"]:
#         bond_types.add( bond["type"] )
# %%
