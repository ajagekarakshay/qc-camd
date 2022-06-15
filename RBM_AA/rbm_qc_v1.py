import numpy as np
import matplotlib.pyplot as plt
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from time import time
import dimod
from minorminer import find_embedding
import sys
import pickle

nprs = np.random.RandomState(12345)

#np.seterr(all='raise')
#np.seterr(under='ignore')

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class QRBM():

    def __init__(self, input, nvis, nhid, solver="Advantage_system1.1", embedding=None, W=None, b=None, c=None):
        self.input = input
        self.nvis = nvis
        self.nhid = nhid
        
        self.nsamples = self.input.shape[0]
        self.Beff = 1.0

        if W is None:
            W = nprs.uniform(low=-1, high=1, size=(self.nvis, self.nhid))
        if b is None:
            b = nprs.uniform(low=-1, high=1, size=self.nvis)
        if c is None:
            c = nprs.uniform(low=-1, high=1, size=self.nhid)

        self.W = W
        self.b = b
        self.c = c

        self.hist = {'ce_loss':[]}
        self.sampler = DWaveSampler(solver = solver)
        if embedding is None:
            embedding = self.get_embedding(self.generate_bqm())
        self.embedding = embedding
        self.system = FixedEmbeddingComposite(self.sampler, embedding=self.embedding)

    def generate_bqm(self):
        lin = {}
        qua = {}

        for i in range(self.nvis):
            lin['v',i] = -1 * self.b[i]

        for j in range(self.nhid):
            lin['h',j] = -1 * self.c[j]

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

        for sample, energy, num, cf in response.data():
            samples.append(sample)
            energies.append(energy)
            num_occur.append(num)

        return np.array(samples), np.array(energies), np.array(num_occur)
    
    def post_process(self, samples, energies, num_occur):

        nsamples = []
        nenergies = []
        nnum_occur = []

        for s in len(samples):
            pass
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

    def weight_update(self, lr=0.1, alpha=1):
        vh_mean, v_mean, h_mean = self.compute_averages()

        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        #print(vh_mean)
        self.W = alpha * self.W + lr * (np.dot(self.input.T, ph_sample) / self.nsamples - vh_mean)
        self.b = alpha * self.b + lr * (np.mean(self.input, axis=0) - v_mean)
        self.c = alpha * self.c + lr * (np.mean(ph_sample, axis=0) - h_mean)

        #if np.max(self.W) > 50:
        #    print("Gone wrong")

    def train(self, epochs, lr=0.1, alpha=1):

        for epoch in range(epochs):
            print("Epoch : ", epoch)
            ce_loss = self.cross_entropy()
            print("Loss : ", ce_loss)
            self.hist['ce_loss'].append(ce_loss)
            self.weight_update(lr=lr, alpha=alpha)
            
            

    def sample_h_given_v(self, v0):
        h1_mean = sigmoid(np.dot(v0, self.W) + self.c)
        h1_sample = nprs.binomial(size=h1_mean.shape, n=1, p=h1_mean)
    
        return h1_mean, h1_sample
    
    def sample_v_given_h(self, h0):
        v1_mean = sigmoid(np.dot(h0, self.W.T) + self.b)
        v1_sample = nprs.binomial(size=v1_mean.shape, n=1, p=v1_mean)
    
        return v1_mean, v1_sample

    def reconstruct(self, v):
        h =  sigmoid(np.dot(v, self.W) + self.c)
        v_recons = sigmoid(np.dot(h, self.W.T) + self.b)
        return v_recons

    def cross_entropy(self):
        hsig = sigmoid(np.dot(self.input, self.W) + self.c)
        vsig = sigmoid(np.dot(hsig, self.W.T) + self.b)

        loss = -np.mean(np.sum(self.input * np.log(vsig) + (1-self.input) * np.log(1 - vsig), axis=1))
        return loss

#fp = open("data\\rbm\\500_6.txt", "rb")
#data = pickle.load(fp)
#fp.close()#

#data2 = np.random.randint(low=0, high=2, size=(500, 52))
#qrbm = QRBM(data2, 52, 52)

#qrbm.train(20)

#recon = qrbm.reconstruct(data)
#recon = 1 * (recon > 0.5)
def plot(n):
    side = int(data[n].shape[0] ** 0.5)

    fig, ax = plt.subplots(2)
    plt.gray()

    ax[0].imshow(data[n].reshape(side, side))
    ax[1].imshow(recon[n].reshape(side, side))
      
    plt.show()

def plot2(data, recon, low, high):
    fig, ax = plt.subplots(2)
    plt.gray()

    ax[0].imshow(data[low:high, :])
    ax[1].imshow(recon[low:high, :])

    plt.show()

def test(epochs):
    fp = open("data\\rbm\\500_6.txt", "rb")
    data = pickle.load(fp)
    fp.close()

    #data2 = np.random.randint(low=0, high=2, size=(500, 52))
    
    qrbm = QRBM(data, 6, 6)

    qrbm.train(epochs)

    recon = qrbm.reconstruct(data)
    recon = 1 * (recon >= 0.5)

    return qrbm, recon