import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

# Discriminative Conditional RBM with support for gaussian visible unit
# Auxilarry connections between condition vector and hidden can be turned ON/FF
# VIS is labels

class DCRBM(nn.Module):
    def __init__(self, ncond, nvis, nhid, 
                 W=None, b=None, c=None, A=None, B=None, 
                 auxillary=True, vis_type="unspecified",
                 sample=False):

        super().__init__()
        self.nvis = nvis
        self.nhid = nhid
        self.ncond = ncond
        self.vis_type = vis_type
        self.auxillary = auxillary
        self.sample = sample

        if W is None:
            W = nn.Parameter(torch.randn(nvis, nhid) * 0.1) 
        if b is None:
            b = nn.Parameter(torch.zeros(1,self.nvis))
        if c is None:
            c = nn.Parameter(torch.zeros(1,self.nhid))
        if A is None:
            A = nn.Parameter(torch.randn(self.ncond,self.nvis) * 0.1) 
        if B is None and auxillary:
            B = nn.Parameter(torch.randn(self.ncond,self.nhid) * 0.1) 
        else:
            B = torch.zeros((self.ncond, self.nhid))

        self.W = W
        self.b = b
        self.c = c
        self.A = A
        self.B = B

    def sample_h_given_vu(self, v, u_cond):
        pre_activation = torch.matmul(v, self.W) + torch.matmul(u_cond, self.B) + self.c
        h1_mean = torch.sigmoid( pre_activation )
        h1_sample = torch.bernoulli(h1_mean)
        return h1_mean.detach(), h1_sample.detach()

    def sample_v_given_hu(self, h, u_cond):
        pre_activation = torch.matmul(h, self.W.t()) + torch.matmul(u_cond, self.A) + self.b
        if self.vis_type == "gaussian":
            raise NotImplementedError
        elif self.vis_type == "binary": #Vis is one-hot label
            v1_mean = torch.exp( pre_activation )
            v1_mean = F.normalize(v1_mean, p=1, dim=1)
            max_idx = torch.argmax(v1_mean, 1)
            one_hot = F.one_hot(max_idx, num_classes=self.nvis).float()
            return v1_mean.detach(), one_hot.detach()
        else:
            raise ValueError(f"Vis unit type ({self.vis_type}) not defined")

    def sample_v_given_u(self, u_cond):
        precomputed_factor = torch.matmul(u_cond, self.B) + self.c
        class_probabilities = torch.zeros((u_cond.shape[0], self.nvis))
        additive_term = torch.matmul(u_cond, self.A)

        for y in range(self.nvis):
            prod = torch.zeros(u_cond.shape[0])
            prod += self.b[0,y]
            prod += additive_term[:,y]
            for j in range(self.nhid):
                prod += torch.log(1 + torch.exp(precomputed_factor[:,j] + self.W[y, j]))
            class_probabilities[:, y] = prod

        copy_probabilities = torch.zeros(class_probabilities.shape)

        for c in range(self.nvis):
          for d in range(self.nvis):
            copy_probabilities[:, c] += torch.exp(-1*class_probabilities[:, c] + class_probabilities[:, d])

        copy_probabilities = 1/copy_probabilities


        class_probabilities = copy_probabilities
        max_idx = torch.argmax(class_probabilities, 1)
        one_hot = F.one_hot(max_idx, num_classes=self.nvis)

        return class_probabilities, one_hot

    def gibbs_hvh(self, h, u_cond):
        v1_mean, v1_sample = self.sample_v_given_hu(h, u_cond)
        h1_mean, h1_sample = self.sample_h_given_vu(v1_sample, u_cond)
        return v1_mean, v1_sample, h1_mean, h1_sample

    def cdk(self, visible_data, cond_data, lr=0.05, k=1, momentum=1, negative_grads=False):
        batch_size = len(visible_data)

        # positive phase
        ph_mean, ph_sample = self.sample_h_given_vu(visible_data, cond_data)

        
        ph_multiplier = ph_sample if self.sample else ph_mean


        vh_data = torch.matmul(visible_data.t(), ph_multiplier)
        uh_data = torch.matmul(cond_data.t(), ph_multiplier)
        uv_data = torch.matmul(cond_data.t(), visible_data)

        # Negative phase
        chain_start = ph_sample
        for step in range(k):
            if step==0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start, cond_data)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples, cond_data)

        nh_multipler = nh_samples if self.sample else nh_means

        vh_model = torch.matmul(nv_samples.t(), nh_multipler)
        uh_model = torch.matmul(cond_data.t(), nh_multipler)
        uv_model = torch.matmul(cond_data.t(), nv_samples)

        W_grad = (vh_data-vh_model) / batch_size                   ############ UPdate rules should include real-valued activation probs and NOT samples 
                                                                ################(Restricted Boltzmann Machine Derivations: https://ofai.at/papers/oefai-tr-2014-13.pdf)
        b_grad = (visible_data-nv_samples).sum(dim=0) / batch_size
        c_grad = (ph_sample-nh_samples).sum(dim=0) / batch_size   ###### Changed from Github
        A_grad = (uv_data-uv_model) / batch_size
        B_grad = (uh_data-uh_model) / batch_size if self.auxillary else None

        #print(W_grad, b_grad, c_grad, A_grad, B_grad)

        self.update_weights(W_grad, b_grad, c_grad, A_grad, B_grad=B_grad, momentum=momentum, lr=lr)


    def update_weights(self, W_grad, b_grad, c_grad, A_grad, B_grad=None, momentum=1., lr=0.05):
        self.W.data = momentum*self.W.data + lr*W_grad
        self.b.data = momentum*self.b.data + lr*b_grad
        self.c.data = momentum*self.c.data + lr*c_grad
        self.A.data = momentum*self.A.data + lr*A_grad
        if self.auxillary:
            self.B.data = momentum*self.B.data + lr*B_grad

    def free_energy(self, v, u_cond):
        wx_b = torch.matmul(v, self.W) + torch.matmul(u_cond, self.B) + self.c
        hidden_term = torch.log(1 + torch.exp(wx_b)).sum(dim=1)
        ax_b = torch.matmul(u_cond, self.A) + self.b
        if self.vis_type=="gaussian":
            visible_term = torch.sum(0.5*torch.square(v-ax_b), dim=1)
        elif self.vis_type=="binary":
            visible_term = torch.sum( -torch.mul(ax_b, v), dim=1)
        else:
            raise ValueError(f"Vis unit type ({self.vis_type}) not defined")

        fe = visible_term - hidden_term
        return fe.mean()


    def compute_loss_metric(self, visible_data, cond_data):
        h1_mean, h1_sample = self.sample_h_given_vu(visible_data, cond_data)
        v1_mean, v1_sample = self.sample_v_given_hu(h1_sample, cond_data)

        fe_vis = self.free_energy(visible_data, cond_data)
        fe_diff =  fe_vis - self.free_energy(v1_sample, cond_data)

        return fe_vis, fe_diff

    def compute_prediction_metric(self, visible_data, cond_data):
        probs, vis_pred = self.sample_v_given_u(cond_data)
        ce_loss = -torch.log( (probs * visible_data).sum(dim=1) ).mean()
        accuracy = (vis_pred*visible_data).sum(dim=1).mean()
        return ce_loss, accuracy

    def forward(self, visible_data, cond_data):
        h1_mean, h1_sample = self.sample_h_given_vu(visible_data, cond_data)
        v1_mean, v1_sample = self.sample_v_given_hu(h1_sample, cond_data)
        return v1_sample


    def predict_cond(self, vis, mu=0, var=1, lazy=True, posterior=None):
        batch_size = len(vis)
        if lazy:
            target_space = torch.linspace(-2, 2, 400)
            if posterior is None:
                fes = [ [self.free_energy( v.view(1,-1), t.view(1,-1) ) for t in target_space] \
                        for v in vis ]
            else:
                #[mu, var] = posterior
                fes = [ [self.free_energy( v.view(1,-1), t.view(1,-1) ) + (t-mu)**2/(2*var) \
                         for t in target_space] for v in vis ]
            fes = torch.Tensor(fes)
            target = target_space[fes.argmin(dim=1)].view(batch_size, -1)
        else:
            pass
        return target


class DCRBM_pl(pl.LightningModule):
    def __init__(self, transform, mode="cd", **kwargs):
        super().__init__()
        self.transform = transform
        self.automatic_optimization = False
        self.rbm = DCRBM(nvis=kwargs["nvis"], 
                          nhid=kwargs["nhid"], 
                          ncond=kwargs["ncond"],
                          auxillary=kwargs.get("auxillary", True),
                          vis_type=kwargs.get("vis_type", "binary"),
                          sample=kwargs.get("sample", False)
                          )
        self.lr = kwargs.get("lr", 0.001)
        self.momentum = kwargs.get("momentum", 1)
        self.mode = mode
        self.kwargs = kwargs

    def get_inputs(self, x):
        raise NotImplementedError

    def forward(self, x): # Graphs as input
        x = self.transform(x)
        u_cond, vis = self.get_inputs(x)  
        return self.rbm(vis, u_cond)

    def training_step(self, x, batch_idx):
        x = self.transform(x)
        u_cond, vis = self.get_inputs(x)

        fe_vis, fe_diff = self.rbm.compute_loss_metric(vis, u_cond)
        ce_loss, accuracy = self.rbm.compute_prediction_metric(vis, u_cond)

        if self.mode == "cd":
            self.rbm.cdk(vis, u_cond,
                        lr=self.lr, 
                        momentum=self.momentum,
                        ) # Train step

        elif self.mode == "disc":
            opt = self.optimizers()
            opt.zero_grad()
            loss = ce_loss
            self.manual_backward(loss)
            opt.step()

        elif self.mode == "hybrid":
            self.rbm.cdk(vis, u_cond,
                        lr=self.lr, 
                        momentum=self.momentum,
                        )
            
            opt = self.optimizers()
            opt.zero_grad()
            loss = ce_loss
            self.manual_backward(loss)
            opt.step()

        else:
            raise ValueError("Mode must be cd, fe or hybrid")

        
        self.log("train/fe", fe_vis,  on_step=False, on_epoch=True)
        self.log("train/fe_diff", fe_diff,  on_step=False, on_epoch=True)
        self.log("train/cross_entropy", ce_loss,  on_step=False, on_epoch=True)
        self.log("train/accuracy", accuracy,  on_step=False, on_epoch=True)


    def validation_step(self, x, batch_index):
        x = self.transform(x)
        u_cond, vis = self.get_inputs(x)
    
        fe_vis, fe_diff = self.rbm.compute_loss_metric(vis, u_cond)
        ce_loss, accuracy = self.rbm.compute_prediction_metric(vis, u_cond)
        
        self.log("val/fe", fe_vis,  on_step=False, on_epoch=True)
        self.log("val/fe_diff", fe_diff,  on_step=False, on_epoch=True)
        self.log("val/cross_entropy", ce_loss,  on_step=False, on_epoch=True)
        self.log("val/accuracy", accuracy,  on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.rbm.parameters(), lr=self.lr)

    def test_step(self, x, batch_idx):
        x = self.transform(x)
        u_cond, vis = self.get_inputs(x)
        
        ce_loss, accuracy = self.rbm.compute_prediction_metric(vis, u_cond)
    
        self.log("test/accuracy", accuracy,  on_step=False, on_epoch=True)

        return accuracy