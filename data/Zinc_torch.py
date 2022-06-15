import os
import os.path as osp
import shutil
import pickle
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                extract_tar)
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

class Zinc_mod(InMemoryDataset):

    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(self, root, transform=None,
                 pre_transform=None, pre_filter=None):
     
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, 'zinc.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            '250k_rndm_zinc_drugs_clean_3.csv', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        #name = 'subset' if self.subset else 'full'
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['zinc.pt']

    def download(self):
        #print(self.raw_dir)
        #shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.raw_dir)
        #extract_tar(path, self.raw_dir)
        #os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.processed_dir)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            from rdkit.Chem import Descriptors
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

        types = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 
                 'S': 4, 'Cl': 5, 'Br': 6, 'I':7,
                 'P': 8} 
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        mols = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]))

        data_list = []
        for index, item in mols.iterrows():
            skip = False
            try:
                mol = Chem.MolFromSmiles(item["smiles"].rstrip())
            except:
                print("Skipped molecule")
                continue
            
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                try:
                    type_idx.append(types[atom.GetSymbol()])
                except KeyError:
                    print("Skipped molecule with ", atom.GetSymbol())
                    skip = True
                    break
            if skip: continue

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x = x.to(torch.float)

            #y = torch.tensor([item["mwt"], item["reactive"], item["logp"]], dtype=torch.float)
            y = torch.tensor([ item["qed"], item["SAS"], item["logP"] ], dtype=torch.float)
            y = y.unsqueeze(0)
            #name = item["zinc_id"]
            name = item["smiles"].rstrip()

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name, idx=index,
                        mol=mol)

            if self.pre_filter is not None and not self.pre_filter(data):
                    continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list),
                        osp.join(self.processed_dir, 'zinc.pt'))
    

class Zinc_loader(pl.LightningDataModule):
    def __init__(self, data, subset, batch_size=20):
        super().__init__()
        assert subset, "Loader can only load a subset of molecules"
        self.data = data
        self.batch_size = batch_size
        self.slices = {}

        splits = ["train", "test", "val"]
        for split in splits:
            with open(osp.join(data.processed_dir, f"{split}.index"), "r") as f:
                idx = [int(x) for x in f.read()[:-1].split(',')]
            self.slices[split] = np.array(idx)
        
    def train_dataloader(self):
        return DataLoader(self.data[self.slices["train"]], batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.data[self.slices["test"]], batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.data[self.slices["val"]], batch_size=self.batch_size, num_workers=8)