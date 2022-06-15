import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
import dimod
from minorminer import find_embedding

nprs = np.random.RandomState(12345)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class QCRBM():

    def __init__(self, input, input_history, nvis, nhid, delay=2, W=None, vbias=None, hbias=None, A=None, B=None, vis_type="binary", solver="DW_2000Q_6"):

        self.input = input
        self.input_history = input_history
        self.nvis = nvis
        self.nhid = nhid
        self.delay = delay
        self.nsamples = self.input.shape[0]
        self.vis_type = vis_type

        if W is None:
            W = np.round(np.asarray(0.01 * nprs.randn(nvis, nhid)),3)
        self.W = W

        if vbias is None:
            vbias = np.zeros(nvis)
        self.vbias = vbias

        if hbias is None:
            hbias = np.zeros(nhid)
        self.hbias = hbias

        if A is None:
            A = np.round(np.asarray(0.01 * nprs.randn(nvis * delay, nvis)),3)
        self.A = A

        if B is None:
            B = np.round(np.asarray(0.01 * nprs.randn(nvis * delay, nhid)),3)
        self.B = B

        self.hist = {'ce_loss': [self.cross_entropy(self.input, self.input_history)], 'mse_loss': [self.mse_loss(self.input, self.input_history)],
                    'fe_loss': [self.free_energy(self.input, self.input_history)],
                    'cost': [self.get_cost(self.input, self.input_history)] }
        # self.hist = {'ce_loss': [], 'mse_loss': [],
        #             'fe_loss': [], 'cost': [] }

        self.sampler = DWaveSampler(solver = solver)
        self.embedding = self.get_embedding(self.generate_bqm()) 
        self.system = FixedEmbeddingComposite(self.sampler, embedding=self.embedding)
        self.timing = None

    # from Git code   
    def free_energy1(self, v_sample, v_history): # for gaussian vis units
        wx_b = np.dot(v_sample, self.W) + np.dot(v_history, self.B) + self.hbias
        ax_b = np.dot(v_history, self.A) + self.vbias
        visible_term = np.sum(0.5 * np.square(v_sample-ax_b))
        hidden_term = np.sum(np.log(1+np.exp(wx_b)))
        return visible_term - hidden_term

    def free_energy(self, v_sample, v_history):  # FOr binary inputs case only
        wx_b = np.dot(v_sample, self.W) + np.dot(v_history, self.B) + self.hbias
        hidden_term = np.sum(np.log(1+np.exp(wx_b)))
        visible_term = - np.sum(np.dot(v_sample, self.vbias)) - np.sum( np.multiply(np.dot(v_history, self.A), v_sample) )
        return visible_term - hidden_term

    def cross_entropy(self, v_sample, v_history):
        pre_activation = np.dot(v_sample, self.W) + np.dot(v_history, self.B) + self.hbias
        hsig = sigmoid(pre_activation)
        pre_activation = np.dot(hsig, self.W.T) + np.dot(v_history, self.A) + self.vbias
        vsig = sigmoid(pre_activation)
        loss = -np.mean(np.sum(v_sample * np.log(vsig) + (1-v_sample) * np.log(1 - vsig), axis=1))
        return loss
    
    def mse_loss(self, v_sample, v_history):
        pre_activation = np.dot(v_sample, self.W) + np.dot(v_history, self.B) + self.hbias
        hsig = sigmoid(pre_activation)
        pre_activation = np.dot(hsig, self.W.T) + np.dot(v_history, self.A) + self.vbias
        vsig = sigmoid(pre_activation)
        loss = np.sum((v_sample-vsig)**2)
        return loss

    def get_cost(self, v_sample, v_history):
        _, _, _, v1_sample = self.gibbs_vhv(v_sample, v_history)
        cost = self.free_energy(v_sample, v_history) - self.free_energy(v1_sample, v_history)
        return cost

    def sample_h_given_v(self, v0, v_history):
        pre_activation = np.dot(v0, self.W) + np.dot(v_history, self.B) + self.hbias
        h1_mean = sigmoid(pre_activation)
        h1_sample = nprs.binomial(size=h1_mean.shape, n=1, p=h1_mean)

        return h1_mean, h1_sample

    def sample_v_given_h(self, h0, v_history):
        pre_activation = np.dot(h0, self.W.T) + np.dot(v_history, self.A) + self.vbias
        v1_mean = sigmoid(pre_activation)

        if self.vis_type == "binary":
            v1_sample = nprs.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        elif self.vis_type == "gaussian":
            v1_sample = pre_activation

        return v1_mean, v1_sample

    def gibbs_hvh(self, h0, v_history):
        v1_mean, v1_sample = self.sample_v_given_h(h0, v_history)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample, v_history)
        return v1_mean, v1_sample, h1_mean, h1_sample

    def gibbs_vhv(self, v0, v_history):
        h1_mean, h1_sample = self.sample_h_given_v(v0, v_history)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample, v_history)
        return h1_mean, h1_sample, v1_mean, v1_sample

    def cdk(self, visible_data, condition_data, lr=0.1, k=1, momentum=1):
        ph_mean, ph_sample = self.sample_h_given_v(visible_data, condition_data)
        chain_start = ph_sample
        for step in range(k):
            if step==0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start, condition_data)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples, condition_data)
        
        batch_samples = visible_data.shape[0]

        self.W = self.W * momentum + lr * (np.dot(visible_data.T, ph_sample)  - np.dot(nv_samples.T, nh_samples)) / batch_samples
        self.vbias = self.vbias * momentum + lr * np.mean(visible_data - nv_samples, axis=0)
        self.hbias = self.hbias * momentum + lr * np.mean(ph_sample - nh_samples, axis=0)         
        self.A = self.A * momentum + lr * (np.dot(condition_data.T, visible_data) - np.dot(condition_data.T, nv_samples)) / batch_samples
        self.B = self.B * momentum + lr * (np.dot(condition_data.T, ph_sample) - np.dot(condition_data.T, nh_samples)) / batch_samples

        self.hist['fe_loss'].append(self.free_energy(visible_data, condition_data))
        self.hist['ce_loss'].append(self.cross_entropy(visible_data, condition_data))
        self.hist['mse_loss'].append(self.mse_loss(visible_data, condition_data))
        self.hist['cost'].append(self.get_cost(visible_data,condition_data))

    def predict(self, v_history, v_init, num_gibbs = 5):
        for i in range(num_gibbs):
            hmean, hsample, vmean, vsample = self.gibbs_vhv(v_init, v_history)
            v_init = vsample
        # mean field approx
        vmean, vsample = self.sample_v_given_h(hmean, v_history)
        return vsample, hsample


    def generate_bqm(self):
        lin = {}
        qua = {}

        for i in range(self.nvis):
            lin['v',i] = -1 * self.vbias[i]

        for j in range(self.nhid):
            lin['h',j] = -1 * self.hbias[j]

        for i in range(self.nvis):
            for j in range(self.nhid):
                qua[('v',i), ('h',j)] = -1 * self.W[i,j]

        bqm = dimod.BinaryQuadraticModel(lin, qua, 0.0, dimod.BINARY)
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

        return vh_mean, v_mean, h_mean

    def weight_update(self, visible_data, condition_data, lr=0.1, momentum=1):
        vh_mean, v_mean, h_mean = self.compute_averages()

        ph_mean, ph_sample = self.sample_h_given_v(visible_data, condition_data)

        batch_samples = visible_data.shape[0]

        self.W = self.W * momentum + lr * ( np.dot(visible_data.T, ph_sample) / batch_samples  - vh_mean ) 
        self.vbias = self.vbias * momentum + lr * ( np.mean(visible_data, axis=0) - v_mean )
        self.hbias = self.hbias * momentum + lr * ( np.mean(ph_sample, axis=0) - h_mean )         
        self.A = self.A * momentum + lr * ( np.dot(condition_data.T, visible_data / batch_samples - v_mean) ) 
        self.B = self.B * momentum + lr * ( np.dot(condition_data.T, ph_sample / batch_samples - h_mean) )

        self.hist['fe_loss'].append(self.free_energy(visible_data, condition_data))
        self.hist['ce_loss'].append(self.cross_entropy(visible_data, condition_data))
        self.hist['mse_loss'].append(self.mse_loss(visible_data, condition_data))
        self.hist['cost'].append(self.get_cost(visible_data,condition_data))


    # new update - Working
    def qcdk(self, visible_data, condition_data, lr=0.1, k=1, momentum=1):

        ph_mean, ph_sample = self.sample_h_given_v(visible_data, condition_data)

        batch_samples = visible_data.shape[0]
        high_samples = False
        if batch_samples > 10000: batch_samples=10000; high_samples = True

        samples, energies, num_occur = self.qsample(num_reads=batch_samples)

        #assert batch_samples == len(num_occur)
        assert batch_samples == sum(num_occur)

        nv_samples = np.zeros((batch_samples, self.nvis))
        nh_samples = np.zeros((batch_samples, self.nhid))
        
        # for i in range(batch_samples):
        #     sam = samples[i]

        #     for j in range(self.nvis):
        #         nv_samples[i,j] = sam['v',j]
        #     for k in range(self.nhid):
        #         nh_samples[i,k] = sam['h',k]
        count = 0
        for i in range(len(num_occur)):
            sam = samples[i]
            for _ in range(num_occur[i]):
                for j in range(self.nvis):
                    nv_samples[count,j] = sam['v',j]
                for k in range(self.nhid):
                    nh_samples[count,k] = sam['h',k]
                count += 1

        if high_samples:
            condition_mean = np.mean(condition_data, axis=0)
            visible_mean = np.mean(visible_data, axis=0)
            hidden_mean = np.mean(ph_sample, axis=0)
            visible_sample_mean = np.mean(nv_samples, axis=0)
            hidden_sample_mean = np.mean(nh_samples, axis=0)

            self.W = self.W * momentum + lr * (visible_mean.reshape(-1,1) @ hidden_mean.reshape(1,-1) - visible_sample_mean.reshape(-1,1) @ hidden_sample_mean.reshape(1,-1))
            self.vbias = self.vbias * momentum + lr * (visible_mean - visible_sample_mean)
            self.hbias = self.hbias * momentum + lr * (hidden_mean - hidden_sample_mean)
            self.A = self.A * momentum + lr * condition_mean.reshape(-1,1) @ (visible_mean-visible_sample_mean).reshape(1,-1)
            self.B = self.B * momentum + lr * condition_mean.reshape(-1,1) @ (hidden_mean-hidden_sample_mean).reshape(1,-1)

        else:
            self.W = self.W * momentum + lr * (np.dot(visible_data.T, ph_sample)  - np.dot(nv_samples.T, nh_samples)) / batch_samples
            self.vbias = self.vbias * momentum + lr * np.mean(visible_data - nv_samples, axis=0)
            self.hbias = self.hbias * momentum + lr * np.mean(ph_sample - nh_samples, axis=0)         
            self.A = self.A * momentum + lr * (np.dot(condition_data.T, visible_data) - np.dot(condition_data.T, nv_samples)) / batch_samples
            self.B = self.B * momentum + lr * (np.dot(condition_data.T, ph_sample) - np.dot(condition_data.T, nh_samples)) / batch_samples

        self.hist['fe_loss'].append(self.free_energy(visible_data, condition_data))
        self.hist['ce_loss'].append(self.cross_entropy(visible_data, condition_data))
        self.hist['mse_loss'].append(self.mse_loss(visible_data, condition_data))
        self.hist['cost'].append(self.get_cost(visible_data,condition_data))