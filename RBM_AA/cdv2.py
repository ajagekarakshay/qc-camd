import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle

nprs = np.random.RandomState(12345)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class RBM():

    def __init__(self, nvis, nhid, W=None, b=None, c=None):
        self.input = input
        self.nvis = nvis
        self.nhid = nhid

        if W is None:
            a = 1.0 / nvis
            W = np.round(nprs.uniform(low=-1, high=1, size=(nvis,nhid)), 2)
        
        if b is None:
            b = np.round(nprs.uniform(low=-1, high=1, size=nvis), 2)
        if c is None:
            c = np.round(nprs.uniform(low=-1, high=1, size=nhid), 2)

        self.W = W
        self.b = b
        self.c = c

        # self.hist = {'ce_loss': [self.cross_entropy(self.input)], 
        #              'mse_loss': [self.mse_loss(self.input)],
        #              'fe_loss': [self.free_energy(self.input)]}

        self.hist = {'ce_loss': [], 
                      'mse_loss': [],
                      'fe_loss': []}

    def sample_h_given_v(self, v0):
        h1_mean = sigmoid(np.dot(v0, self.W) + self.c)
        h1_sample = nprs.binomial(size=h1_mean.shape, n=1, p=h1_mean)

        return h1_mean, h1_sample

    def sample_v_given_h(self, h0):
        v1_mean = sigmoid(np.dot(h0, self.W.T) + self.b)
        v1_sample = nprs.binomial(size=v1_mean.shape, n=1, p=v1_mean)

        return v1_mean, v1_sample

    def gibbs_hvh(self, h0):
        v1_mean, v1_sample = self.sample_v_given_h(h0)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return v1_mean, v1_sample, h1_mean, h1_sample

    def cdk(self, visible_data, lr=0.1, k=1, momentum=1, negative_grads=False):
        ph_mean, ph_sample = self.sample_h_given_v(visible_data)
        chain_start = ph_sample

        for step in range(k):
            if step==0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        batch_samples = visible_data.shape[0]

        Gw = (np.dot(visible_data.T, ph_sample)  - np.dot(nv_samples.T, nh_samples)) / batch_samples
        Gb = np.mean(visible_data - nv_samples, axis=0)
        Gc = np.mean(ph_sample - nh_samples, axis=0)

        if negative_grads:
            Gw = -Gw
            Gb = -Gb
            Gc = -Gc
            
        self.W = self.W * momentum + lr * Gw
        self.b = self.b * momentum + lr * Gb
        self.c = self.c * momentum + lr * Gc

        # self.hist['ce_loss'].append( self.cross_entropy(visible_data) )
        # self.hist['mse_loss'].append( self.mse_loss(visible_data) )
        # self.hist['fe_loss'].append( self.free_energy(visible_data) )

    def reconstruct(self, v):
        h =  sigmoid(np.dot(v, self.W) + self.c)
        v_recons = sigmoid(np.dot(h, self.W.T) + self.b)
        return v_recons

    def cross_entropy(self, v_sample):
        hsig = sigmoid(np.dot(v_sample, self.W) + self.c)
        vsig = sigmoid(np.dot(hsig, self.W.T) + self.b)

        loss = -np.mean(np.sum(v_sample * np.log(vsig) + (1-v_sample) * np.log(1 - vsig), axis=1))
        return loss
    
    def mse_loss(self, v_sample):
        hsig = sigmoid(np.dot(v_sample, self.W) + self.c)
        vsig = sigmoid(np.dot(hsig, self.W.T) + self.b)

        loss = np.sum((v_sample-vsig)**2)
        return loss

    def free_energy(self, v_sample):  # FOr binary inputs case only
        wx_b = np.dot(v_sample, self.W) + self.c
        hidden_term = np.sum(np.log(1+np.exp(wx_b)))
        visible_term = - np.sum(np.dot(v_sample, self.b))
        return visible_term - hidden_term

    def train(self, epochs=10, lr=0.1):
        for i in range(epochs):
            ce_loss = self.cross_entropy()
            print("Epoch : ", i, " Cross entropy : ", ce_loss)
            self.hist['ce_loss'].append(ce_loss)
            
            self.cdk(lr=lr, k=1)




# Train example

#data = np.array([[1,1,1,0,0,0],
#                    [1,0,1,0,0,0],
#                    [1,1,1,0,0,1],
#                    [0,0,1,1,1,0],
#                    [0,0,1,1,0,0],
#                    [0,0,1,1,1,0]])


#fp = open("data\\rbm\\500_6.txt", "rb")
#data = pickle.load(fp)
#fp.close()

#data2 = np.random.randint(low=0, high=2, size=(500,52))
#rbm = RBM(data2, 52, 52)

#epochs = 50
#ce = []
#mse = []

#for epoch in range(epochs):
#    print("Epoch : ", epoch, " CE Loss : ", rbm.cross_entropy(), "MSE loss : ", rbm.mse_loss())
#    rbm.cdk(lr=0.1, k=1)
#    
#    ce.append(rbm.cross_entropy())
#    mse.append(rbm.mse_loss())


#v = np.array([[1, 1, 1, 0, 0, 0],
#                [0, 0, 1, 1, 1, 0],
#                [0, 0, 0, 1, 1, 0],
#                [1, 1, 0, 0, 0, 0]])

#recon = rbm.reconstruct(data)
#recon = 1 * (recon > 0.5)
#print(vrecons > 0.5)

#plt.plot(ce)
#plt.title("Training")
#plt.xlabel("Epochs")
#plt.ylabel("Cross entropy")
#plt.show()

def plot(n):
    side = int(data[n].shape[0] ** 0.5)

    fig, ax = plt.subplots(2)
    ax[0].imshow(data[n].reshape(side, side))
    ax[1].imshow(recon[n].reshape(side, side))
    plt.gray()
    plt.show()


def plot2(low, high):
    fig, ax = plt.subplots(2)
    plt.gray()

    ax[0].imshow(data[low:high, :])
    ax[1].imshow(recon[low:high, :])

    plt.show()