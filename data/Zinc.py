import os
import os.path as osp
import pickle

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import tensorflow as tf

from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot
from spektral.utils.io import load_csv, load_sdf

class Zinc():
    def __init__(self, split="train"):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.path = osp.join(self.raw_path, f'{split}.pickle')
    
    def read(self):
        print("Loading Zinc dataset..")
        with open(self.path, "rb") as fp:
            mols = pickle.load(fp)
        
        


    @property
    def raw_path(self):
        return "data/Zinc/raw/"
    
    @property
    def processed_path(self):
        return "data/Zinc/processed/"