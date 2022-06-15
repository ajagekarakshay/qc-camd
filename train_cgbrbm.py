# %%
import torch
import pytorch_lightning as pl
import pickle
import torch.nn.functional as F

from RBM_AA.cgbrbm_torch import CGBRBM_pl

from data.transform import Binning, NeuralFP_TD, NormalizeTarget, NeuralFP_continous

import os, wandb
os.environ["WANDB_API_KEY"] = "29ef10a563126c993b31f306e5a17f55b237e430"
wandb.login()


from data.Zinc_torch import Zinc_mod, Zinc_loader

target_names = {0:"qed", 1:"sas", 2:"logP"}
target_idx = [0,1,2] # 0:qed, 1:sas, 2:logP


target_transform = Binning("models/zinc_eq_freq_bins_10_qed_sas_logP.npz",
                           target_idx=target_idx
                           )

data = Zinc_mod(root="data/Zinc/",
               transform=target_transform
               )

loader = Zinc_loader(data, subset=True, batch_size=10_000)
print("\n### Data loading complete ###\n")


#fp_transform = NeuralFP("checkpoints/fpnn/fpnn_large_v3.pt")

#fp_transform = NeuralFP_continous("models/gnn_fp_continous_256.pt")

fp_transform = NeuralFP_TD("models/td_neural_fp.pt")

#### config
mode = "cd"

version = "v1"
auxillary = True

#name = f"{mode}_{target_names[target_idx]}_{version}_aux_{auxillary}"
name = f"{mode}_multi_{version}_aux_{auxillary}"


lr = 0.001
momentum = 1

#nvis = data[0].target.size(-1)      ##########previous
#ncond = transform(data[0]).fp.size(-1)

def get_inputs(x):
   ucond, vis = x.fp, x.label
   return ucond, vis

def get_dims():
   x = fp_transform(data[0])
   ucond, vis = get_inputs(x)
   ncond = ucond.size(-1)
   nvis = vis.size(-1)
   nhid = 64
   return ncond, nvis, nhid


ncond, nvis, nhid = get_dims()
print(f"Dims : {ncond}, {nvis}, {nhid}")

#### config-end

if mode == "qc":
   from RBM_AA.qc_crbm_torch import QC_CGBRBM_pl
   rbm = QC_CGBRBM_pl(fp_transform, nvis=nvis, nhid=nhid, ncond=ncond,
              lr=lr, momentum=momentum,
              mode=mode,
              #sample=True
              auxillary=auxillary,
              vis_type = "binary" ######### was oreviously gaussian
              )
else:
   rbm = CGBRBM_pl(fp_transform, nvis=nvis, nhid=nhid, ncond=ncond,
              lr=lr, momentum=momentum,
              mode=mode,
              #sample=True
              auxillary=auxillary,
              vis_type = "binary" ######### was oreviously gaussian
              )  



rbm.get_inputs = get_inputs

config = {
            "nvis":nvis,
            "nhid":nhid,
            "ncond":ncond,
            "lr": lr,
            "momentum": momentum
         }


logger = pl.loggers.WandbLogger(
                                project="qc-camd-v2", 
                                group=f"CGBRBM_{name}",
                                config=config,
                                log_model=True
                                )

run_name = logger.experiment.name

checkpoint = pl.callbacks.ModelCheckpoint(
                                          dirpath=f"checkpoints/cgbrbm/{name}/{run_name}",
                                          monitor="val/fe_diff",
                                          mode="min"
                                          )

trainer = pl.Trainer(
                    logger=logger, 
                    max_epochs=200, 
                    callbacks=[checkpoint])

trainer.fit(rbm, loader)

#results = trainer.test(rbm, dataloaders=loader.test_dataloader(),
#                        ckpt_path="best")