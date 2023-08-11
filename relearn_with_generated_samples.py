from rbm import *
import numpy
import matplotlib.pyplot as plt
import os

LOAD_FILE = os.path.dirname(os.path.abspath(__file__)) + '/results/220507075839/mpf/min_kl_train_8.npz'
BURN_IN = 100
N = 1000000
SKIP = 1
COL = 10
ROW = N // COL

GPU = True  #Only used for calculating KL divergence, not training
OPTIMIZER = 'adam'
OPT_PARAMS = {'lr':0.005, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':0.999997}
EPOCH = 1000
BATCH = 100
HIDDEN = 20
CD_K_LIST = [1, 2, 4, 8, 16]
SEED = 1
CALC_KL_STEP = 20

if __name__ == '__main__':
    rbm = RBM(0, 0) #nv and nh are automatically set when loading
    rbm.load(LOAD_FILE)
    
    v = np.zeros((1, rbm.nv))
    vs = []
    hs = []
    for i in range(BURN_IN + N * SKIP):
        h, ph = rbm.get_hidden(v)
        v, pv = rbm.get_visible(h)
        if i >= BURN_IN:
            if (i - BURN_IN) % SKIP == 0:
                vs.append(v[0])
                hs.append(h[0])

    vs = np.array(vs)
    hs = np.array(hs)
    print(rbm.calc_kl_divergence(vs, duplicate=True))
    print(rbm.calc_kl_divergence_all_visible(vs, hs, duplicate=True))

    rbm = RBM(rbm.nv, rbm.nh, optimizer=OPTIMIZER, opt_params=OPT_PARAMS, batch_size=BATCH, gpu=GPU)

    for epoch in range(EPOCH):
        if epoch % CALC_KL_STEP == 0:
            kl = rbm.calc_kl_divergence_all_visible(vs, hs, duplicate=True)
            print(f"Epoch:{epoch} kl:{kl}")
        rbm.train_cd_all_visible(vs, hs, 1)
    