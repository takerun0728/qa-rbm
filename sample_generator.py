from rbm import *
import numpy
import matplotlib.pyplot as plt
import os

LOAD_FILE = os.path.dirname(os.path.abspath(__file__)) + '/results/220507075839/mpf/min_kl_train_8.npz'
BURN_IN = 100
N = 50
SKIP = 100
COL = 10
ROW = N // COL

if __name__ == '__main__':
    rbm = RBM(0, 0) #nv and nh are automatically set when loading
    rbm.load(LOAD_FILE)
    
    v = np.zeros((1, rbm.nv))
    pvs = []
    for i in range(BURN_IN + N * SKIP):
        h, ph = rbm.get_hidden(v)
        v, pv = rbm.get_visible(h)
        if i >= BURN_IN:
            if (i - BURN_IN) % SKIP == 0:
                pvs.append(pv)
    
    pvs = np.array(pvs)
    pvs = pvs.reshape(N, -1)
    pvs = np.insert(pvs, 0, np.zeros(N), axis=1)
    pvs = np.insert(pvs, 5, np.zeros(N), axis=1)
    pvs = np.insert(pvs, 30, np.zeros(N), axis=1)
    pvs = np.insert(pvs, 35, np.zeros(N), axis=1)
    pvs = pvs.reshape(N, 6, 6)

    orgs_raw = np.load(os.path.dirname(os.path.abspath(__file__)) + '/processed_mnist.npy')
    orgs = np.insert(orgs_raw, 0, np.zeros(orgs_raw.shape[0]), axis=1)
    orgs = np.insert(orgs, 5, np.zeros(orgs.shape[0]), axis=1)
    orgs = np.insert(orgs, 30, np.zeros(orgs.shape[0]), axis=1)
    orgs = np.insert(orgs, 35, np.zeros(orgs.shape[0]), axis=1)
    orgs = orgs.reshape(-1, 6, 6)

    #Display samples of training data
    fig = plt.figure()
    for i, org in enumerate(orgs[0:N]):
        ax = fig.add_subplot(ROW, COL, i + 1)
        ax.imshow(org)
    plt.show()

    #Display generated images from the trained model
    fig = plt.figure()
    for i, pv in enumerate(pvs):
        ax = fig.add_subplot(ROW, COL, i + 1)
        ax.imshow(pv)
    plt.show()
    
    fig = plt.figure()
    #Display reconstructed images
    for i, org in enumerate(orgs_raw[:COL]):
        h, ph = rbm.get_hidden(org.reshape(1, -1))
        v, pv = rbm.get_visible(h)
        org = np.insert(org, 0, 0)
        org = np.insert(org, 5, 0)
        org = np.insert(org, 30, 0)
        org = np.insert(org, 35, 0)
        org = org.reshape(6, 6)
        pv = np.insert(pv, 0, 0, axis=1)
        pv = np.insert(pv, 5, 0, axis=1)
        pv = np.insert(pv, 30, 0, axis=1)
        pv = np.insert(pv, 35, 0, axis=1)
        pv = pv.reshape(6, 6)
        ax = fig.add_subplot(2, COL, i + 1)
        ax.imshow(org)
        ax = fig.add_subplot(2, COL, i + COL + 1)
        ax.imshow(pv)
    plt.show()

    