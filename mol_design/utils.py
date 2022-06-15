from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, QED
#import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from .sasscorer import *

atom_types = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 
              'S': 4, 'Cl': 5, 'Br': 6, 'I':7,
              'P': 8}

valency = {'C': 4, 'O': 2, 'N': 3, 'F': 1, 
              'S': 2, 'Cl': 1, 'Br': 1, 'I': 1,
              'P': 3
            }

atom_type_inv = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 
            4: 'S', 5: 'Cl', 6: 'Br', 7:'I',
            8: 'P'}

color_map = {'C': 'grey',
             'O': 'orange',
             'N': 'green',
             'F':'cyan',
             'S':'yellow',
             "Cl":"pink"
             } 

edge_type = {0:Chem.rdchem.BondType.SINGLE, 
             1:Chem.rdchem.BondType.DOUBLE,
             2:Chem.rdchem.BondType.TRIPLE}

def data_to_mol(X, A, E=None):
    X = X.numpy() if type(X) is not np.ndarray else X
    idx = np.argmax(X, axis=1)
    atom_sym = [atom_type_inv[i] for i in idx]
    mol = Chem.MolFromSmiles(atom_sym[0])
    for i in range(1,len(atom_sym)):
        mol = Chem.CombineMols(mol, Chem.MolFromSmiles(atom_sym[i]))
    mol = Chem.EditableMol(mol)
    start, end = np.where(A==1)
    for row, col in zip(start,end):
        if col > row:
            bond_type = edge_type[0] if E is None else edge_type[E[row,col]]
            mol.AddBond(int(row), int(col), order=bond_type)
    mol = mol.GetMol()
    mol = AllChem.RemoveHs(mol)
    return mol

def data_to_nx(X, A):
    X = X.numpy() if type(X) is not np.ndarray else X
    G = nx.Graph()
    idx = np.argmax(X, axis=1)
    atom_sym = [atom_type_inv[i] for i in idx]
    for i, atom in enumerate(atom_sym):
        G.add_node(i, atom_symbol=atom)
    start, end = np.where(A==1)
    for row, col in zip(start,end):
        if col > row:
            G.add_edge(row, col)
    return G

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def edges_to_matrix(edge_idx, Natoms):
    if type(edge_idx) is not np.ndarray: edge_idx = edge_idx.numpy()
    A = np.zeros((Natoms, Natoms))
    rows, cols = edge_idx
    A[rows, cols] = 1
    return A

def draw_nx_graph(ng, node_size=600, **kwargs):
    atom_sym = nx.get_node_attributes(ng, 'atom_symbol')
    #print(atom_sym)
    colors = [color_map[atom_sym[atom]] for atom in atom_sym]
    node_colors = [atom_types[atom_sym[atom]] for atom in atom_sym]
    #print(node_colors)
    obj = nx.draw(ng, labels=atom_sym,
                  with_labels=True,
                  #node_color=colors,
                  node_color= node_colors,#range(ng.number_of_nodes()),
                  node_size=node_size,
                  cmap=plt.get_cmap("Set1"),
                  **kwargs
                  )
    return obj

def edge_matrix_validity(x, A: np.ndarray, fully_connected=False):
    types = x.argmax(dim=1).numpy()
    types = [atom_type_inv[i] for i in types]
    assert is_symmetric(A), "A matrix should be symmetric"
    bonds = np.sum(A, axis=1)

    if not fully_connected:
        validities = [bonds[i] <= valency[j] for i,j in enumerate(types)]
    else:
        validities = [ 1 <= bonds[i] and bonds[i] <= valency[j] for i,j in enumerate(types)]

    Nviolations = len(validities) - np.count_nonzero(validities)
    valid = np.prod(validities)
    return valid, validities

def is_symmetric(a: np.ndarray):
    return (a == a.T).all()

def xA_to_graph(x:torch.Tensor, A:np.ndarray):
    row, col = np.nonzero(A)
    edge_index = torch.from_numpy( np.array([row, col]) )
    mol = data_to_mol(x, A)
    qed = QED.qed(mol)
    logP = Crippen.MolLogP(mol)
    sas = calculate_sas_score(mol)
    prop = {"qed":qed, "sas":sas, "logP":logP}
    name = Chem.MolToSmiles(mol)
    return Data(x=x, edge_index=edge_index, mol=mol, prop=prop, name=name)