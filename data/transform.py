import numpy as np
np.seterr(all="ignore")
import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from models.fp import FPNN_v3
from torchdrug.models import NeuralFingerprint
from torchdrug.data import Graph
from torch_geometric.data.batch import Batch

target_names = {0:"qed", 1:"sas", 2:"logP"}

class Binning:
    def __init__(self, path_to_bins, target_idx:list):
        self.target_idx = target_idx
        bin_data = np.load(path_to_bins)
        
        self.bins = [bin_data[target_names[idx]] for idx in target_idx]
        self.nbin = [len(self.bins[i])-1 for i in range(len(target_idx))]
        
    def __call__(self, data):
        # Labeling starts from 1
        labels = [np.digitize(data.y[:,idx], self.bins[i]) - 1 for i,idx in enumerate(self.target_idx)]
        labels = [np.clip(label, 0, self.nbin[i]-1) for i,label in enumerate(labels)]
        labels = [F.one_hot(torch.as_tensor(label).to(torch.int64), 
                              num_classes=self.nbin[i]).float() for i,label in enumerate(labels)]
        
        data.label = torch.cat( (labels), dim=1 )
        return data


class NeuralFP_TD:
    def __init__(self, path):
        self.fpnn = NeuralFingerprint(input_dim=9, output_dim=512, 
                                        hidden_dims=100)
        data = torch.load(path)
        self.fpnn.load_state_dict( data.state_dict() )

    def __call__(self, data):
        if isinstance(data, Batch):
            num_graphs = data.num_graphs
            fps = [ self.fpnn( Graph(edge_list=data[i].edge_index.t()), data[i].x )['graph_feature'].view(-1) for i in range(num_graphs) ]
            fps = torch.stack(fps, dim=0)
        else:
            fps = [ self.fpnn( Graph(edge_list=data.edge_index.t()), data.x )['graph_feature'].view(-1) ]
            fps = torch.stack(fps, dim=0)
        data.fp = fps
        return data

# class NeuralFP:
#     def __init__(self, path_to_fpnn):
#         self.fpnn = torch.load(path_to_fpnn).eval()
#     def __call__(self, data):
#         fp = self.fpnn.fingerprint(data)
#         data.fp = torch.from_numpy(fp).float()
#         return data

class NeuralFP_continous:
    def __init__(self, path_to_fpnn):
        #self.fpnn = torch.load(path_to_fpnn).eval()
        data = torch.load(path_to_fpnn)
        fpnn = FPNN_v3(9, out_channels=256, dense_size=512)
        fpnn.load_state_dict( data.state_dict() )
        self.fpnn = fpnn
    def __call__(self, data):
        fp = self.fpnn.fingerprint(data)
        data.fp = torch.from_numpy(fp).float()
        return data

class MorganFP:
    def __init__(self):
        pass
    def __call__(self, data):
        mols = data.mol if type(data.mol) is list else [data.mol]
        fp = [AllChem.GetMorganFingerprintAsBitVect(i, 2, nBits=512) for i in mols]
        fp = np.array( [fp_to_binary_vector(s) for s in fp] )
        data.fp = torch.from_numpy(fp).float()
        return data

class MaccsFP:
    def __init__(self):
        pass
    def __call__(self, data):
        mols = data.mol if type(data.mol) is list else [data.mol]
        fp = [MACCSkeys.GenMACCSKeys(i) for i in mols]
        fp = np.array( [fp_to_binary_vector(s) for s in fp] )
        data.fp = torch.from_numpy(fp).float()
        return data
        

class NormalizeTarget:
    def __init__(self, path_to_scaler, target_idx = [0,1,2], transform=True):
        self.scaler = pickle.load(open(path_to_scaler, "rb"))
        self.target_idx = target_idx
        self.transform = transform
    def __call__(self, data):
        target = data.y
        target = self.scaler.transform(target) [:, self.target_idx] if self.transform \
                    else data.y[:, self.target_idx]
        data.target = torch.from_numpy(target).float() if self.transform else target
        return data


# Convert Rdkit fingerprint to numpy array
def fp_to_binary_vector(fp):
    s = fp.ToBitString()
    return np.fromstring(s,'u1') - ord('0')