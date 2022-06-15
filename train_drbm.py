# %%
import torch
import pytorch_lightning as pl
import pickle

from RBM_AA.drbm_torch import DRBM_pl
from data.transform import Binning, NeuralFP

import os, wandb
os.environ["WANDB_API_KEY"] = "29ef10a563126c993b31f306e5a17f55b237e430"
wandb.login()


from data.Zinc_torch import Zinc_mod, Zinc_loader

data = Zinc_mod(root="data/Zinc/",
               transform=Binning("models/zinc_bins_freq_10.npz")
               )
loader = Zinc_loader(data, subset=True, batch_size=1000)
print("\n### Data loading complete ###\n")


transform = NeuralFP("checkpoints/fpnn/fpnn_large_v3.pt")

nvis = transform(data[0]).fp.size(-1)
nclass = data[0].label.size(-1)
nhid = 64


name = "cd"

rbm = DRBM_pl(transform, nvis=nvis, nhid=nhid, nclass=nclass,
              lr=0.05, momentum=1,
              mode=name,
              sample=True)   ###### turn this off for default behaviour

config = {
            "nvis":nvis,
            "nhid":nhid,
            "nclass":nclass,
            "lr": 0.05,
            "momentum": 1
         }


logger = pl.loggers.WandbLogger(project="qc-camd", 
                                group=f"DRBM_{name}",
                                config=config,
                                log_model=True)

checkpoint = pl.callbacks.ModelCheckpoint(
                                          dirpath=f"checkpoints/drbm_{name}/",
                                          monitor="val/acc",
                                          mode="max")

trainer = pl.Trainer(logger=logger, max_epochs=1000, callbacks=[checkpoint])

trainer.fit(rbm, loader)

results = trainer.test(rbm, dataloaders=loader.test_dataloader())