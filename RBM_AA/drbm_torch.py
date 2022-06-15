import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

#Discriminative RBM Torch

class DRBM(nn.Module):
    def __init__(self, nvis, nhid, nclass, W=None, b=None, c=None, d=None, U=None):
        super().__init__()
        self.nvis = nvis
        self.nhid = nhid
        self.nclass = nclass
        self.loss = nn.CrossEntropyLoss()

        if W is None:
            W = nn.Parameter(torch.randn(nvis, nhid))
        if b is None:
            b = nn.Parameter(torch.randn(1,self.nvis))
        if c is None:
            c = nn.Parameter(torch.randn(1,self.nhid))
        if d is None:
            d = nn.Parameter(torch.randn(1,self.nclass))
        if U is None:
            U = nn.Parameter(torch.randn(self.nclass,self.nhid))

        self.W = W
        self.b = b
        self.c = c
        self.d = d
        self.U = U

        # self.hist = {'ce_loss': [self.cross_entropy(self.input)], 
        #              'mse_loss': [self.mse_loss(self.input)],
        #              'fe_loss': [self.free_energy(self.input)]}
        #self.eval()

        self.hist = {'ce_loss': [], 
                      'mse_loss': [],
                      'fe_loss': []}

    def sample_h_given_vy(self, v0, y0):
        h1_activations = torch.matmul(v0, self.W) +  self.c + torch.matmul(y0, self.U)
        h1_mean = torch.sigmoid( h1_activations )
        h1_sample = torch.bernoulli(h1_mean)
        return h1_mean, h1_sample

    def sample_v_given_h(self, h0):
        v1_mean = torch.sigmoid( torch.matmul(h0, self.W.t()) + self.b ) 
        v1_sample = torch.bernoulli(v1_mean)
        return v1_mean, v1_sample

    def sample_y_given_h(self, h0):
        class_probs = torch.exp( torch.matmul(h0, self.U.t()) + self.d )
        class_probs = F.normalize(class_probs, p=1, dim=1)
        max_idx = torch.argmax(class_probs, 1)
        one_hot = F.one_hot(max_idx, num_classes=self.nclass)
        return class_probs, one_hot.float()

    def sample_y_given_v(self, v0):
        # constant = torch.matmul(v0, self.W) + self.c
        # class_probs = torch.zeros((len(v0), self.nclass)).float()

        # for y in range(self.nclass):
        #     prod_term = 1 + torch.exp( constant + self.U[y] )
        #     prod = torch.prod(prod_term, dim=1)
        #     class_probs[:, y] = prod
        
        # class_probs = torch.exp(self.d) * class_probs
        # class_probs = F.normalize(class_probs, p=1, dim=1)
        # max_idx = torch.argmax(class_probs, 1)
        # one_hot = F.one_hot(max_idx, num_classes=self.nclass)
        # return class_probs, one_hot.float()

        precomputed_factor = torch.matmul(v0, self.W) + self.c
        class_probabilities = torch.zeros((v0.shape[0], self.nclass))

        for y in range(self.nclass):
            prod = torch.zeros(v0.shape[0])
            prod += self.d[0,y]
            for j in range(self.nhid):
                prod += torch.log(1 + torch.exp(precomputed_factor[:,j] + self.U[y, j]))
            #print(prod)
            class_probabilities[:, y] = prod  

        copy_probabilities = torch.zeros(class_probabilities.shape)

        for c in range(self.nclass):
          for d in range(self.nclass):
            copy_probabilities[:, c] += torch.exp(-1*class_probabilities[:, c] + class_probabilities[:, d])

        copy_probabilities = 1/copy_probabilities


        class_probabilities = copy_probabilities
        max_idx = torch.argmax(class_probabilities, 1)
        one_hot = F.one_hot(max_idx, num_classes=self.nclass)

        return class_probabilities, one_hot

    def gibbs_hvh(self, h0):
        v1_mean, v1_sample = self.sample_v_given_h(h0)
        y1_mean, y1_sample = self.sample_y_given_h(h0)

        h1_mean, h1_sample = self.sample_h_given_vy(v1_sample, y1_sample)

        return v1_mean, v1_sample, y1_mean, y1_sample, h1_mean, h1_sample

    def cdk(self, visible_data, y, lr=0.05, k=1, momentum=1, negative_grads=False, factor=1, sample=False):
        batch_size = len(visible_data)

        # positive phase
        ph_mean, ph_sample = self.sample_h_given_vy(visible_data, y)
        vh_data = torch.matmul(visible_data.t(), ph_mean)
        yh_data = torch.matmul(y.t(), ph_mean)

        # Negative phase
        chain_start = ph_sample
        for step in range(k):
            if step==0:
                nv_means, nv_samples, ny_means, ny_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, ny_means, ny_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        if not sample:
            vh_model = torch.matmul(nv_samples.t(), nh_means)
            yh_model = torch.matmul(ny_samples.t(), nh_means)
        else:
            vh_model = torch.matmul(nv_samples.t(), nh_samples)
            yh_model = torch.matmul(ny_samples.t(), nh_samples)

        # Grads
        W_grad = (vh_data-vh_model) / batch_size
        U_grad = (yh_data-yh_model) / batch_size
        b_grad = (visible_data-nv_samples).sum(dim=0) / batch_size
        c_grad = (ph_sample-nh_samples).sum(dim=0) / batch_size   ###### Changed from Github
        d_grad = (y-ny_samples).sum(dim=0) / batch_size

        self.update_weights(W_grad, U_grad, b_grad, c_grad, d_grad, momentum=momentum, lr=lr, factor=factor)

    def disc(self, visible_data, y, lr=0.05, factor=1, opt=torch.optim.SGD):
        actual_labels = torch.argmax(y, dim=1)
        opt = opt(self.parameters(), lr=lr)
        opt.zero_grad()

        class_probs, _ = self.sample_y_given_v(visible_data)
        loss = self.loss(class_probs, actual_labels)
        loss.backward()
        opt.step()
        
    def update_weights(self, W_grad, U_grad, b_grad, c_grad, d_grad, momentum=1., lr=0.05, factor=1.):
        self.W.data = momentum*self.W.data + lr*factor*W_grad
        self.U.data = momentum*self.U.data + lr*factor*U_grad
        self.b.data = momentum*self.b.data + lr*factor*b_grad
        self.c.data = momentum*self.c.data + lr*factor*c_grad
        self.d.data = momentum*self.d.data + lr*factor*d_grad

    def compute_loss_metric(self, visible_data, y):
        class_probs, predicted_labels = self.sample_y_given_v(visible_data)
        predicted_labels = torch.argmax(predicted_labels, dim=1)
        actual_labels = torch.argmax(y, dim=1)
        error = self.loss(class_probs, actual_labels)
        accuracy = torch.mean((predicted_labels == actual_labels)*1.)
        return error, accuracy

    def forward(self, v):
        class_probs, predicted_labels = self.sample_y_given_v(v)
        return class_probs, predicted_labels


class DRBM_pl(pl.LightningModule):
    def __init__(self, transform, mode="cd", **kwargs):
        super().__init__()
        self.transform = transform
        self.automatic_optimization = False
        self.rbm = DRBM(nvis=kwargs["nvis"], nhid=kwargs["nhid"], nclass=kwargs["nclass"])
        self.lr = kwargs["lr"]
        self.momentum = kwargs["momentum"]
        self.mode = mode
        self.kwargs = kwargs

    def forward(self, x): # Graphs as input
        x = self.transform(x)
        return self.rbm(x.fp)

    #def on_fit_start(self):
    #    self.log("train/")

    def training_step(self, x, batch_idx):
        x = self.transform(x)
        vis, y = x.fp, x.label.float()

        if self.mode == "cd":
            self.rbm.cdk(vis, y,
                        lr=self.lr, 
                        momentum=self.momentum,
                        sample=self.kwargs["sample"]) # Train step
        elif self.mode == "disc":
            self.rbm.disc(vis, y,
                          lr=self.lr)
        elif self.mode == "hybrid":
            pass
        else:
            raise ValueError("Mode must be cd, disc or hybrid")

        loss, accuracy = self.rbm.compute_loss_metric(vis, y)
        
        self.log("train/loss", loss,  on_step=False, on_epoch=True)
        self.log("train/acc", accuracy,  on_step=False, on_epoch=True)


    def validation_step(self, x, batch_index):
        x = self.transform(x)
        vis, y = x.fp, x.label.float()
        loss, accuracy = self.rbm.compute_loss_metric(vis, y)
        self.log("val/loss", loss,  on_step=False, on_epoch=True)
        self.log("val/acc", accuracy,  on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return None

    def test_step(self, x, batch_idx):
        x = self.transform(x)
        vis, y = x.fp, x.label.float()
        loss, accuracy = self.rbm.compute_loss_metric(vis, y)
        self.log("test/loss", loss,  on_step=False, on_epoch=True)
        self.log("test/acc", accuracy,  on_step=False, on_epoch=True)
        return loss, accuracy