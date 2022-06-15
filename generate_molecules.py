
import pytorch_lightning as pl
from data.Zinc_torch import Zinc_mod, Zinc_loader
import os, wandb
import sys

from models.prop import Prop_v3
from data.transform import MorganFP, MaccsFP, NormalizeTarget, NeuralFP_continous, Binning

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

from mol_design.qubo_opt import QuboOPT
from mol_design.utils import *
import pickle


target_names = {0:"qed", 1:"sas", 2:"logP"}
target_idx = 2 # 0:qed, 1:sas, 2:logP
target_name = target_names[target_idx]

fp_transform = NeuralFP_continous("models/gnn_fp_continous_256.pt")

target_transform = Binning(
                           "models/zinc_eq_freq_bins_10_qed_sas_logP.npz",
                           target_idx=target_idx
                           )

data = Zinc_mod(
               root="data/Zinc/",
               transform=target_transform
               )

loader = Zinc_loader(data, subset=True, batch_size=10_000)

for btest in loader.test_dataloader():
    break

bins = target_transform.bins
print(bins)

####### Change this here
label_idx = 4
target_label = F.one_hot(torch.Tensor([label_idx]).long(), num_classes=10).float()

lower_lim = bins[label_idx]
upper_lim = bins[label_idx+1]
print(f"Target range : {lower_lim} to {upper_lim}")
print("Target label : ", target_label)

from RBM_AA.cgbrbm_torch import CGBRBM_pl

rbm = CGBRBM_pl(fp_transform, nvis=10, ncond=256, nhid=64)

ckpt_path = f"checkpoints/cgbrbm/cd_{target_name}_v1_aux_True/attentive-etchings-20/epoch=165-step=165.ckpt"
#ckpt_path = f"checkpoints/cgbrbm/cd_{target_name}_v1_aux_True/mesmerizing-admirer-13/epoch=167-step=167.ckpt"

ckpt = torch.load(ckpt_path)

rbm.load_state_dict( ckpt["state_dict"] )



def target_func(x, A):
    graph = xA_to_graph(x, A)
    fp = fp_transform(graph).fp
    fe_vis, fe_diff, recon_error = rbm.rbm.compute_loss_metric(target_label, fp)
    #return fe_diff.detach().numpy()
    return fe_vis.detach().numpy()

def save(obj):
    file_name = f"logs/{target_name}_{label_idx}_cgbrbm.pkl"
    pickle.dump(obj, open(file_name, "wb"))


valid_graphs = []

for i in range(1000):
    qopt = QuboOPT( btest[i].x, target_func=target_func, 
                    uncertainty=False
                    )
    try:
        qopt.minimize(iterations = 90, verbose=False)
    except:
        print("Stopping minimize")

    for gr in qopt.graph_history:
        if gr.prop[target_name] >= lower_lim and gr.prop[target_name] <= upper_lim:
            valid_graphs.append( gr )

    print(f"{i}: {len(valid_graphs)}")


