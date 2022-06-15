import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
import dimod
from minorminer import find_embedding
import pickle

# Conditional RBM with support for gaussian visible unit
# Auxilarry connections between condition vector and hidden can be turned ON/OFF

class QC_CGBRBM(nn.Module):
    def __init__(self, ncond, nvis, nhid, 
                 W=None, b=None, c=None, A=None, B=None, 
                 auxillary=True, vis_type="unspecified",
                 solver="Advantage_system4.1"):

        super().__init__()
        self.nvis = nvis
        self.nhid = nhid
        self.ncond = ncond
        self.vis_type = vis_type
        self.auxillary = auxillary

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

        self.sampler = DWaveSampler(solver = solver)
        try:
            self.embedding = pickle.load(open(f"models/embedding_{nvis}_{nhid}.pkl", "rb"))
            print("Existing embeddign used")
        except:
            self.embedding = self.get_embedding(self.generate_bqm()) 
        self.system = FixedEmbeddingComposite(self.sampler, embedding=self.embedding)
        self.timing = None

    def sample_h_given_vu(self, v, u_cond):
        pre_activation = torch.matmul(v, self.W) + torch.matmul(u_cond, self.B) + self.c
        h1_mean = torch.sigmoid( pre_activation )
        h1_sample = torch.bernoulli(h1_mean)
        return h1_mean.detach(), h1_sample.detach()

    def sample_v_given_hu(self, h, u_cond):
        pre_activation = torch.matmul(h, self.W.t()) + torch.matmul(u_cond, self.A) + self.b
        if self.vis_type == "gaussian":
            v1_mean = pre_activation
            #v1_sample = torch.norma(v1_mean, torch.ones(len(v1_mean)))
            v1_sample = v1_mean
            return v1_mean.detach(), v1_sample.detach()
        elif self.vis_type == "binary":
            v1_mean = torch.sigmoid(pre_activation)
            v1_sample = torch.bernoulli(v1_mean)
            return v1_mean.detach(), v1_sample.detach()
        else:
            raise ValueError(f"Vis unit type ({self.vis_type}) not defined")

    def gibbs_hvh(self, h, u_cond):
        v1_mean, v1_sample = self.sample_v_given_hu(h, u_cond)
        h1_mean, h1_sample = self.sample_h_given_vu(v1_sample, u_cond)
        return v1_mean, v1_sample, h1_mean, h1_sample

    def generate_bqm(self):
        lin = {}
        qua = {}

        for i in range(self.nvis):
            lin['v',i] = -1 * self.b[0, i]

        for j in range(self.nhid):
            lin['h',j] = -1 * self.c[0, j]

        for i in range(self.nvis):
            for j in range(self.nhid):
                qua[('v',i), ('h',j)] = -1 * self.W[i,j]

        bqm = dimod.BinaryQuadraticModel(lin, qua, 0.0, dimod.BINARY)

        #### U_cond not considered here for now
        return bqm

    def get_embedding(self, bqm):
        __, target_edgelist, target_adjacency = self.sampler.structure
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
        embedding = find_embedding(source_edgelist, target_edgelist)
        return embedding

    def qsample(self, num_reads=1000):
        samples = []
        energies = []
        num_occur = []

        response = self.system.sample(self.generate_bqm(), num_reads=num_reads)
        self.timing = response.info

        for sample, energy, num, cf in response.data():
            samples.append(sample)
            energies.append(energy)
            num_occur.append(num)

        return np.array(samples), np.array(energies), np.array(num_occur)
    
    def compute_averages(self):
        samples, energies, num_occur = self.qsample()


        vh_mean = np.zeros((self.nvis, self.nhid))
        v_mean = np.zeros(self.nvis)
        h_mean = np.zeros(self.nhid)

        #N = np.sum(num_occur)
        #P = num_occur

        energies /= np.max(np.abs(energies))
        Z = np.sum(np.exp(-1 * energies))
        N = 1
        P = np.exp(-1 * energies) / Z

        for s in range(len(samples)):

            for i in range(self.nvis):
                for j in range(self.nhid):
                    vh_mean[i,j] += samples[s]['v', i] * samples[s]['h', j] * P[s]

            for i in range(self.nvis):
                v_mean[i] += samples[s]['v',i] * P[s]

            for j in range(self.nhid):
                h_mean[j] += samples[s]['h', j] * P[s]

        vh_mean = vh_mean / N
        v_mean = v_mean / N
        h_mean = h_mean / N

        vh_mean = torch.from_numpy(vh_mean).float()
        v_mean = torch.from_numpy(v_mean).float().view(1, self.nvis)
        h_mean = torch.from_numpy(h_mean).float().view(1, self.nhid)

        return vh_mean, v_mean, h_mean

    def qcdk(self, visible_data, cond_data, lr=0.05, k=1, momentum=1, negative_grads=False):
        batch_size = len(visible_data)

        # positive phase
        ph_mean, ph_sample = self.sample_h_given_vu(visible_data, cond_data)

        # vh_data = torch.matmul(visible_data.t(), ph_mean)
        # uh_data = torch.matmul(cond_data.t(), ph_mean)
        # uv_data = torch.matmul(cond_data.t(), visible_data)

        vh_data = torch.matmul(visible_data.t(), ph_sample)
        uh_data = torch.matmul(cond_data.t(), ph_sample)
        uv_data = torch.matmul(cond_data.t(), visible_data)

        # Negative phase
        
        vh_model, v_model, h_model = self.compute_averages()
        
        uv_model = torch.einsum("nk, zj -> znkj", cond_data, v_model)
        uv_model = uv_model.sum(dim=[0,1]) # consistent with next equations
        
        uh_model = torch.einsum("nk, zj -> znkj", cond_data, h_model)
        uh_model = uh_model.sum(dim=[0,1]) # consistent with next equations

        W_grad = (vh_data-vh_model) / batch_size                   ############ UPdate rules should include real-valued activation probs and NOT samples 
                                                                ################(Restricted Boltzmann Machine Derivations: https://ofai.at/papers/oefai-tr-2014-13.pdf)
        b_grad = (visible_data-v_model).sum(dim=0) / batch_size
        c_grad = (ph_sample-h_model).sum(dim=0) / batch_size   ###### Changed from Github
        A_grad = (uv_data-uv_model) / batch_size
        B_grad = (uh_data-uh_model) / batch_size if self.auxillary else None

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
        recon_error = nn.MSELoss()(visible_data, v1_mean)
        return fe_vis, fe_diff, recon_error

    def forward(self, visible_data, cond_data):
        h1_mean, h1_sample = self.sample_h_given_vu(visible_data, cond_data)
        v1_mean, v1_sample = self.sample_v_given_hu(h1_sample, cond_data)
        return v1_sample

    def predict_target(self, cond_data, niter=50, lazy=False, version="v1"):
        batch_size = len(cond_data)
        if not lazy:
            target = torch.nn.Parameter(torch.randn(batch_size, self.nvis))
            optim = torch.optim.SGD([target], lr=0.1)
            for _ in range(niter):
                optim.zero_grad()
                #fe, fe_diff, recon = self.compute_loss_metric(target, cond_data)
                fe = self.free_energy(target, cond_data)
                loss = fe#_diff
                loss.backward()
                optim.step()
        else:
            target_space = torch.linspace(-2, 2, 400)
            if version == "v2":
                fes = [ [self.free_energy(t.view(1,-1), cond.view(1,-1)) for t in target_space] \
                        for cond in cond_data ]
            else:
                fes = [ [self.free_energy(t.view(1,-1), cond.view(1,-1)) for t in target_space] \
                        for cond in cond_data ]
            fes = torch.Tensor(fes)
            target = target_space[fes.argmin(dim=1)].view(batch_size, -1)

        return target

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

class QC_CGBRBM_pl(pl.LightningModule):
    def __init__(self, transform, mode="cd", **kwargs):
        super().__init__()
        self.transform = transform
        self.automatic_optimization = False
        self.rbm = QC_CGBRBM(nvis=kwargs["nvis"], 
                          nhid=kwargs["nhid"], 
                          ncond=kwargs["ncond"],
                          auxillary=kwargs["auxillary"],
                          vis_type=kwargs["vis_type"]
                          )
        self.lr = kwargs["lr"]
        self.momentum = kwargs["momentum"]
        self.mode = mode
        self.kwargs = kwargs

    def forward(self, x): # Graphs as input
        x = self.transform(x)
        #u_cond, vis = x.fp, x.target        ####### Previous p(y|f)
        u_cond, vis = x.target, x.fp        ###### p(f|y)
        return self.rbm(vis, u_cond)

    def training_step(self, x, batch_idx):
        x = self.transform(x)
        #u_cond, vis = x.fp, x.target ####### Previous p(y|f)
        u_cond, vis = x.target, x.fp

        _, fe_diff, _ = self.rbm.compute_loss_metric(vis, u_cond)

        if self.mode == "qc":
            self.rbm.qcdk(vis, u_cond,
                        lr=self.lr, 
                        momentum=self.momentum,
                        ) # Train step
        elif self.mode == "fe":
            opt = self.optimizers()
            opt.zero_grad()
            loss = fe_diff
            self.manual_backward(loss)
            opt.step()

        elif self.mode == "hybrid":
            self.rbm.cdk(vis, u_cond,
                        lr=self.lr, 
                        momentum=self.momentum,
                        )
            
            opt = self.optimizers()
            opt.zero_grad()
            loss = fe_diff
            self.manual_backward(loss)
            opt.step()

        else:
            raise ValueError(f"Mode {self.mode} not supported")

        fe_vis, fe_diff, recon_error = self.rbm.compute_loss_metric(vis, u_cond)
        
        self.log("train/fe", fe_vis,  on_step=False, on_epoch=True)
        self.log("train/fe_diff", fe_diff,  on_step=False, on_epoch=True)
        self.log("train/recon", recon_error,  on_step=False, on_epoch=True)


    def validation_step(self, x, batch_index):
        x = self.transform(x)
        #u_cond, vis = x.fp, x.target
        u_cond, vis = x.target, x.fp
        fe_vis, fe_diff, recon_error = self.rbm.compute_loss_metric(vis, u_cond)
        
        self.log("val/fe", fe_vis,  on_step=False, on_epoch=True)
        self.log("val/fe_diff", fe_diff,  on_step=False, on_epoch=True)
        self.log("val/recon", recon_error,  on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.rbm.parameters(), lr=self.lr)

    def test_step(self, x, batch_idx):
        x = self.transform(x)
        #u_cond, vis = x.fp, x.target
        u_cond, vis = x.target, x.fp
        
        u_cond_pred = self.rbm.predict_cond(vis, posterior=True)
        
        l1_loss = F.l1_loss(u_cond_pred, u_cond)
        
        self.log("test/l1_loss", l1_loss,  on_step=False, on_epoch=True)

        return l1_loss