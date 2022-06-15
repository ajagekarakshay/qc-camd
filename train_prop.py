# %%
from locale import normalize
import pytorch_lightning as pl
from data.Zinc_torch import Zinc_mod, Zinc_loader
import os, wandb
import sys

from models.prop import Prop_v3
from data.transform import NeuralFP, MorganFP, MaccsFP, NormalizeTarget, NeuralFP_continous

normalize_on = False
version = "v1"

fp_type = sys.argv[1]
name = f"{fp_type}_norm_{normalize_on}"

target_names = {0:"qed", 1:"sas", 2:"logP"}
target_idx = [0, 1, 2] # 0:qed, 1:sas, 2:logP

if fp_type == "neural":
   transform = NeuralFP_continous("models/gnn_fp_continous_256.pt")
elif fp_type == "morgan":
   transform = MorganFP()
elif fp_type == "maccs":
   transform = MaccsFP()
else:
   raise ValueError("Fingerprint not recognized")


os.environ["WANDB_API_KEY"] = "29ef10a563126c993b31f306e5a17f55b237e430"
wandb.login()

data = Zinc_mod(
                root="data/Zinc/", 
                transform=NormalizeTarget(
                              "models/zinc_std_scaler_qed_sas_logp.pkl", 
                              target_idx = target_idx,
                              transform=normalize_on,
                              )
               )        ####### changed

loader = Zinc_loader(data, subset=True, batch_size=10_000)
print("\n### Data loading complete ###\n")


# Load model with fingerprints
input_size = transform(data[0]).fp.size(-1)
print(f"{fp_type} fingerprint size : {input_size}")


loss = "mse"

prop = Prop_v3(transform=transform, input_size=input_size, loss=loss, 
               num_targets=len(target_idx))

logger = pl.loggers.WandbLogger(project="qc-camd-v2", 
                                group=f"Prop_{name}_{version}",
                                #config=config,
                                log_model=True)

run_name = logger.experiment.name

checkpoint = pl.callbacks.ModelCheckpoint(
                                          dirpath=f"checkpoints/prop/{name}/{version}/{run_name}",
                                          monitor=f"val/{loss}",
                                          mode="min"
                                          )

trainer = pl.Trainer(logger=logger, max_epochs=75, callbacks=[checkpoint])

trainer.fit(prop, loader)

results = trainer.test(prop, dataloaders=loader.test_dataloader(), 
                       ckpt_path="best") # tests on best model and not current model