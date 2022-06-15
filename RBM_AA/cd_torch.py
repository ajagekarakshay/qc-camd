import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

#torch.set_default_tensor_type(torch.float32)
#torch.manual_seed(12345)

class RBM(nn.Module):
    def __init__(self, nvis, nhid, W=None, b=None, c=None):
        super().__init__()
        self.nvis = nvis
        self.nhid = nhid

        if W is None:
            W = nn.Parameter(torch.randn(nvis, nhid))
        if b is None:
            b = nn.Parameter(torch.randn(1,self.nvis))
        if c is None:
            c = nn.Parameter(torch.randn(1,self.nhid))

        self.W = W
        self.b = b
        self.c = c

        # self.hist = {'ce_loss': [self.cross_entropy(self.input)], 
        #              'mse_loss': [self.mse_loss(self.input)],
        #              'fe_loss': [self.free_energy(self.input)]}
        #self.eval()

        self.hist = {'ce_loss': [], 
                      'mse_loss': [],
                      'fe_loss': []}

    def sample_h_given_v(self, v0):
        if type(v0) is np.ndarray:
            v0 = torch.from_numpy(v0).float()
        h1_mean = torch.sigmoid( F.linear(v0, self.W.t(), self.c) )
        h1_sample = torch.bernoulli(h1_mean)

        return h1_mean, h1_sample

    def sample_v_given_h(self, h0):
        v1_mean = torch.sigmoid( F.linear(h0, self.W, self.b) )
        v1_sample = torch.bernoulli(v1_mean)

        return v1_mean, v1_sample

    def gibbs_hvh(self, h0):
        v1_mean, v1_sample = self.sample_v_given_h(h0)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return v1_mean, v1_sample, h1_mean, h1_sample

    def cdk(self, visible_data, lr=0.1, k=1, momentum=1, negative_grads=False):
        if type(visible_data) is np.ndarray:
            visible_data = torch.from_numpy(visible_data).float()

        ph_mean, ph_sample = self.sample_h_given_v(visible_data)
        chain_start = ph_sample

        for step in range(k):
            if step==0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        batch_samples = visible_data.shape[0]

        Gw = (torch.matmul(visible_data.t(), ph_sample) - torch.matmul(nv_samples.t(), nh_samples)) / batch_samples
        Gb = torch.mean(visible_data - nv_samples, dim=0, keepdim=True)
        Gc = torch.mean(ph_sample - nh_samples, dim=0, keepdim=True)

        if negative_grads:
            Gw = -Gw
            Gb = -Gb
            Gc = -Gc

        with torch.no_grad():    
            self.W.data = self.W.data * momentum + lr * Gw
            self.b.data = self.b.data * momentum + lr * Gb
            self.c.data = self.c.data * momentum + lr * Gc

        # self.hist['ce_loss'].append( self.cross_entropy(visible_data) )
        # self.hist['mse_loss'].append( self.mse_loss(visible_data) )
        # self.hist['fe_loss'].append( self.free_energy(visible_data) )

    def forward(self, v):
        h, _ =  self.sample_h_given_v(v)
        v_recons, _ = self.sample_v_given_h(h)
        return v_recons

    def cross_entropy(self, v_sample):
        if type(v_sample) is np.ndarray:
            v_sample = torch.from_numpy(v_sample).float()
        hsig, _ = self.sample_h_given_v(v_sample)
        vsig, _ = self.sample_v_given_h(hsig)
        loss = F.binary_cross_entropy(vsig, v_sample) #, reduction="sum")
        return loss
    
    def mse_loss(self, v_sample):
        if type(v_sample) is np.ndarray:
            v_sample = torch.from_numpy(v_sample).float()
        hsig, _ = self.sample_h_given_v(v_sample)
        vsig, _ = self.sample_v_given_h(hsig)

        loss = F.mse_loss(vsig, v_sample) #, reduction="sum")
        return loss

    def free_energy(self, v_sample):  # FOr binary inputs case only
        if type(v_sample) is np.ndarray:
            v_sample = torch.from_numpy(v_sample).float()
        wx_c = F.linear(v_sample, self.W.t(), self.c)
        hidden_term = torch.sum(F.softplus(wx_c), dim=1, keepdim=True)
        visible_term = - torch.matmul(v_sample, self.b.t())
        return torch.mean( visible_term - hidden_term )

    def fe_loss(self, v_sample):
        a = self.free_energy(v_sample)
        b = self.free_energy( self(v_sample) )
        return a-b


class RBM_v2(RBM, pl.LightningModule):
    def __init__(self, fpnn, prop, **kwargs):
        #super().__init__()
        super().__init__(**kwargs)
        self.fpnn = fpnn
        self.fpnn.eval() # Weights frozen
        self.prop = prop
        self.prop.eval()
        self.automatic_optimization = False

    def training_step(self, x, batch_idx):
        with torch.no_grad():
            fp = self.fpnn(x)
            preds = self.prop(x)
        out = torch.cat( [fp, preds], dim=1)

        self.cdk(out) # Train step

        with torch.no_grad():
            fe, mse, ce = self.free_energy(out), self.mse_loss(out), self.cross_entropy(out)
        
        self.log("train/fe", fe, on_step=True, on_epoch=True)
        self.log("train/mse", mse, on_step=True, on_epoch=True)
        self.log("train/ce", ce, on_step=True, on_epoch=True)
        #self.log("train")

    def validation_step(self, x, batch_index):
        fp = self.fpnn(x)
        preds = self.prop(x)
        out = torch.cat( [fp, preds], dim=1)
        fe, mse, ce = self.free_energy(out), self.mse_loss(out), self.cross_entropy(out)
        self.log("val/fe", fe, on_step=True, on_epoch=True)
        self.log("val/mse", mse, on_step=True, on_epoch=True)
        self.log("val/ce", ce, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return None

class RBM_v3(pl.LightningModule):
    def __init__(self, fpnn, scaler, **kwargs):
        #super().__init__()
        super().__init__()
        self.fpnn = fpnn
        self.fpnn.eval() # Weights frozen
        self.scaler = scaler
        self.automatic_optimization = False
        self.rbm = RBM(nvis=kwargs["nvis"], nhid=kwargs["nhid"])
        self.lr = kwargs["lr"]
        self.momentum = kwargs["momentum"]

    def input(self, x):
        with torch.no_grad():
            fp = self.fpnn(x)
        targets = x.y[:,-1:]
        scaled_targets = self.scaler.transform(targets)
        scaled_targets = torch.Tensor(scaled_targets)
        inp = torch.cat( [fp, scaled_targets], dim=1).float()
        return inp

    def forward(self, x):
        inp = self.input(x)
        return self.rbm(inp)

    def training_step(self, x, batch_idx):
        inp = self.input(x)

        self.rbm.cdk(inp, lr=self.lr, momentum=self.momentum) # Train step

        with torch.no_grad():
            fe, mse, ce, fediff = self.rbm.free_energy(inp), self.rbm.mse_loss(inp), self.rbm.cross_entropy(inp), self.rbm.fe_loss(inp)
        
        self.log("train/fe", fe,  on_step=False, on_epoch=True)
        self.log("train/mse", mse,  on_step=False, on_epoch=True)
        self.log("train/ce", ce, on_step=False, on_epoch=True)
        self.log("train/fe_diff", fediff, on_step=False, on_epoch=True)

    def validation_step(self, x, batch_index):
        inp = self.input(x)
        fe, mse, ce, fediff = self.rbm.free_energy(inp), self.rbm.mse_loss(inp), self.rbm.cross_entropy(inp), self.rbm.fe_loss(inp)
        self.log("val/fe", fe,  on_epoch=True)
        self.log("val/mse", mse, on_epoch=True)
        self.log("val/ce", ce, on_epoch=True)
        self.log("val/fe_diff", fediff, on_epoch=True)

    def configure_optimizers(self):
        return None