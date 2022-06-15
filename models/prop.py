import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class Prop_v3(pl.LightningModule):
    def __init__(self, transform, input_size=256, loss="mae", num_targets=2, **kwargs):
        super().__init__()
        self.transform = transform
        self.mlp = nn.Sequential(  nn.Linear(input_size, num_targets)
                                    #nn.Linear(input_size, 64),
                                   #nn.ReLU(),
                                   #nn.Linear(64, 2) 
                                 
                                   )
        self.loss = loss
        if loss == "mae":
            self.loss_fn = F.l1_loss
        elif loss == "mse":
            self.loss_fn = F.mse_loss

    def forward(self, x):
        x = self.transform(x)
        x = x.fp  # Fingerprint
        x = self.mlp(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def training_step(self, x, batch_idx):
        target = x.target
        ypred = self(x)
        loss = self.loss_fn(ypred, target)
        self.log(f"train/{self.loss}", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, x, batch_idx):
        target = x.target
        ypred = self(x)
        mae_loss = self.loss_fn(ypred, target)
        self.log(f"val/{self.loss}", mae_loss, on_step=False, on_epoch=True)

    def test_step(self, x, batch_idx):
        target = x.target
        ypred = self(x)
        [qed_mae, sas_mae, logP_mae] = F.l1_loss(ypred, target, reduction="none").mean(dim=0)
        self.log("test/mae_qed", qed_mae, on_step=False, on_epoch=True)
        self.log("test/mae_sas", sas_mae, on_step=False, on_epoch=True)
        self.log("test/mae_logP", logP_mae, on_step=False, on_epoch=True)
        return qed_mae, sas_mae, logP_mae



class Prop_v2(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fpnn = FPNN_v2(*args)
        self.mlp = nn.Sequential(  nn.Linear(kwargs["dense_size"]*2, 64),
                                   nn.ReLU(),
                                   nn.Linear(16, 1) )

    def forward(self, x):
        x = self.fpnn(x)
        x = self.mlp(x)
        return x
    
    def training_step(self, x, batch_idx):
        target = x.y[:,-1].view(-1,1)
        ypred = self(x)
        loss = F.mse_loss(ypred, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, x, batch_idx):
        target = x.y[:,-1].view(-1,1)
        ypred = self(x)
        loss = F.mse_loss(ypred, target)
        mae_loss = F.l1_loss(ypred, target)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/mae", mae_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())



class Prop_v1(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fpnn = FPNN(*args, **kwargs)
        self.mlp = nn.Sequential(  nn.Linear(256, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1) )

    def forward(self, x):
        x = self.fpnn(x)
        x = self.mlp(x)
        return x

    def training_step(self, x, batch_idx):
        target = x.y[:,-1].view(-1,1)
        ypred = self(x)
        loss = F.mse_loss(ypred, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, x, batch_idx):
        target = x.y[:,-1].view(-1,1)
        ypred = self(x)
        loss = F.mse_loss(ypred, target)
        mae_loss = F.l1_loss(ypred, target)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/mae", mae_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())