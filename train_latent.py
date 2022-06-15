from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb, os

from data.Zinc_torch import Zinc_mod, Zinc_loader
from data.transform import NeuralFP_continous, Binning
from RBM_AA.cgbrbm_torch import CGBRBM_pl
from RBM_AA.dcrbm_torch import DCRBM_pl

os.environ["WANDB_API_KEY"] = "29ef10a563126c993b31f306e5a17f55b237e430"
wandb.login()

target_idx = 2 # 0:qed, 1:sas, 2:logP
target_names = {0:"qed", 1:"sas", 2:"logP"}

fp_transform = NeuralFP_continous("models/gnn_fp_continous_256.pt")
target_transform = Binning("models/zinc_eq_freq_bins_10_qed_sas_logP.npz",
                           target_idx=target_idx
                           )

data = Zinc_mod(
                root="data/Zinc/",
                transform=target_transform
               )    

loader = Zinc_loader(data, subset=True, batch_size=10_000)

for btrain in loader.train_dataloader():
    btrain = fp_transform(btrain)
for bval in loader.val_dataloader():
    bval = fp_transform(bval)
for btest in loader.test_dataloader():
    btest = fp_transform(btest)


# Latent variable computation
mode = "cd"
mtype = "dcrbm"
model_path = f"checkpoints/{mtype}/" + f"{mode}_{target_names[target_idx]}_v1_aux_True"
run_name = "fresh-night-43"
saved_epoch = 195

ckpt_path = f"{model_path}/{run_name}/epoch={saved_epoch}-step={saved_epoch}.ckpt"

ckpt =  torch.load(ckpt_path)


nvis = data[0].label.size(-1)
ncond = bval.fp.size(-1)
nhid = 64

if mtype == "cgbrbm":
    rbm = CGBRBM_pl(None, mode=None, 
                nvis=nvis, ncond=ncond, nhid=nhid,
                auxillary=True,
                vis_type="binary")
elif mtype == "dcrbm":
    rbm = DCRBM_pl(None, mode=None, 
                nvis=nvis, ncond=ncond, nhid=nhid,
                auxillary=True,
                vis_type="binary",
                sample=False)

rbm.load_state_dict(ckpt["state_dict"])


# compute latent vars
hm_train, _ = rbm.rbm.sample_h_given_vu(btrain.label, btrain.fp)
hm_val, _ = rbm.rbm.sample_h_given_vu(bval.label, bval.fp)

_, vis_pred = rbm.rbm.sample_v_given_u(btest.fp)
hm_test, _ = rbm.rbm.sample_h_given_vu(vis_pred.float(), btest.fp)


wandb.init(
    project="qc-camd-v2",
    group=f"latent_{mtype}_{mode}_{target_names[target_idx]}",
    name=run_name
)


# define lieanr model
class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(nhid, 1)
    def forward(self, x):
        return self.l1(x)

lmodel = LM()
optim = torch.optim.Adam(lmodel.parameters())


global_step = 0

def trainer_loop(epochs):
    global global_step
    for epoch in range(epochs):
        optim.zero_grad()

        ypred = lmodel(hm_train)
        loss = F.mse_loss(ypred, btrain.y[:, target_idx:target_idx+1])

        with torch.inference_mode():
            ypred_val = lmodel(hm_val)
            val_loss = F.mse_loss(ypred_val, bval.y[:, target_idx:target_idx+1])
        
        loss.backward()
        optim.step()

        wandb.log( {"train_mse":loss, "val_mse":val_loss},  step=global_step)
        global_step +=1 


    # testing
    ypred_test = lmodel(hm_test)
    test_loss = F.l1_loss(ypred_test, btest.y[:, target_idx:target_idx+1])
    wandb.log({"test_l1_loss": test_loss}, step=global_step-1)


epochs = 50
trainer_loop(epochs)