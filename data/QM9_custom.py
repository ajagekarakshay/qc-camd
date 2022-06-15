import os
import os.path as osp

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import tensorflow as tf

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import Descriptors

from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot
from spektral.utils.io import load_csv, load_sdf

ATOMIC_NUMBER = [1, 6, 7, 8, 9]
ATOMS = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
BOND_TYPES = [1, 2, 3, 4]
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

class QM9_mod(Dataset):
    """
    
    #########################
    New target "logP" added (20th target) 
    #######################
    
    The QM9 chemical data set of small molecules.
    In this dataset, nodes represent atoms and edges represent chemical bonds.
    There are 5 possible atom types (H, C, N, O, F) and 4 bond types (single,
    double, triple, aromatic).
    Node features represent the chemical properties of each atom and include:
    - The atomic number, one-hot encoded;
    - The atom's position in the X, Y, and Z dimensions;
    - The atomic charge;
    - The mass difference from the monoisotope;
    The edge features represent the type of chemical bond between two atoms,
    one-hot encoded.
    Each graph has an 19-dimensional label for regression.
    **Arguments**
    - `amount`: int, load this many molecules instead of the full dataset
    (useful for debugging).
    - `n_jobs`: number of CPU cores to use for reading the data (-1, to use all
    available cores).
    
    """

    def __init__(self, raw_path, amount=None, n_jobs=1, **kwargs):
        self.raw_path = raw_path
        self.amount = amount
        self.n_jobs = n_jobs
        
        super().__init__(**kwargs)

    # def download(self):
    #     get_file(
    #         "qm9.tar.gz",
    #         self.url,
    #         extract=True,
    #         cache_dir=self.path,
    #         cache_subdir=self.path,
    #     )
    #     os.remove(osp.join(self.path, "qm9.tar.gz"))

    def read(self):
        print("Loading QM9 dataset..")

        x_list, a_list, e_list, y_list = [], [], [], []

        skip_file = osp.join(self.raw_path, "uncharacterized.txt")
        with open(skip_file, 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        
        sdf_file = osp.join(self.raw_path, "gdb9.sdf")
        #data = load_sdf(sdf_file, amount=self.amount)  # Internal SDF format
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, 
                                   sanitize=False)

        # Load labels
        labels_file = osp.join(self.raw_path, "gdb9.sdf.csv")
        labels = load_csv(labels_file)
        labels = labels.set_index("mol_id").values

        for i, mol in enumerate(tqdm(suppl, ncols=80)):
            try:
                mol.UpdatePropertyCache()
            except:
                continue
                
            if i in skip:
                continue
            if i == self.amount: # Load only specified number of molecules
                break
            
            try:
                logP = Descriptors.MolLogP(mol)
            except:
                print("Exception")
                logP = -999
            
            if logP == -999:
                continue
            
            N = mol.GetNumAtoms()
            a, e, edge_index = mol_to_adj(mol)
            
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(ATOMS[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = np.array(atomic_number)

            row, col = edge_index.numpy()
            hs = (z == 1) * 1
            num_hs = tf.math.unsorted_segment_sum(hs[row], col, num_segments=N).numpy()
            
            x1 = label_to_one_hot(type_idx, list(ATOMS.values()))
            x2 = np.array([atomic_number, aromatic, sp, sp2, sp3, num_hs]).T
            x = np.concatenate((x1, x2), axis=1)

            y = labels[i].ravel()
            y = np.append(y, [logP])
            
            x_list.append(x)
            a_list.append(a)
            e_list.append(e)
            y_list.append(y)

        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, y_list)
        ]



def mol_to_adj(mol):
    N = mol.GetNumAtoms()
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BONDS[bond.GetBondType()]]
    
    edge_index = tf.constant([row, col])
    edge_type = tf.constant(edge_type)
    edge_attr = label_to_one_hot(edge_type, list(BONDS.values()))

    perm = tf.argsort(edge_index[0] * N + edge_index[1])
    edge_index = tf.gather(edge_index, perm, axis=1)
    edge_type = tf.gather(edge_type, perm)
    edge_attr = tf.gather(edge_attr, perm)

    a = sp.csr_matrix((np.ones_like(row), (row,col)))
    return a, edge_attr, edge_index



def split_n_save(dataset, test_size):
    np.random.shuffle(dataset)
    train_set = dataset[:-test_size]
    test_set = dataset[-test_size:]
    print("Train set : ", len(train_set))
    print("Test set : ", len(test_set))
    path = "QM9\\processed\\"
    import pickle
    train_path = osp.join(path, "train.pkl")
    test_path = osp.join(path, "test.pkl")
    with open(train_path, "wb") as fp:
        pickle.dump(train_set, fp)
    with open(test_path, "wb") as fp:
        pickle.dump(test_set, fp)
