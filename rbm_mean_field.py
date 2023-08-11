from rbm import *
import numpy
import matplotlib.pyplot as plt
import os

LOAD_FILE = os.path.dirname(os.path.abspath(__file__)) + '/results/compare_bm/220507075839/cd/min_kl_train_8.npz'
BURN_IN = 10
MCMC_ITER = 10000

if __name__ == '__main__':
    rbm = RBM(0, 0) #nv and nh are automatically set when loading
    rbm.load(LOAD_FILE)
    ave_v1, ave_h1 = rbm.get_mean_field(n=5)

    v = np.zeros(rbm.nv)
    vs = []
    hs = []
    for i in range(MCMC_ITER + BURN_IN):
        h, _ = rbm.get_hidden(v)
        v, _ = rbm.get_visible(h)
        if i >= BURN_IN:
            vs.append(v)
            hs.append(h)
    ave_v2 = np.average(vs, axis=0)
    ave_h2 = np.average(hs, axis=0)

    print(ave_v1)
    print(ave_h1)

    print(ave_v1)
    print(ave_h1)

    print(ave_v1 - ave_v2)
    print(np.sqrt(np.average((ave_v1 - ave_v2)**2)))
    print(ave_h1 - ave_h2)
    print(np.sqrt(np.average((ave_h1 - ave_h2)**2)))
    