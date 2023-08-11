from rbm import *
import numpy
import matplotlib.pyplot as plt
import os

LOAD_FILE = os.path.dirname(os.path.abspath(__file__)) + '/results/220507075839/mpf/min_kl_train_8.npz'
BURN_IN = 100
N = [1000, 10000, 100000, 1000000]
SKIP = 1
COL = 10

if __name__ == '__main__':
    rbm = RBM(0, 0) #nv and nh are automatically set when loading
    rbm.load(LOAD_FILE)

    sampler_gibbs = RBMSampler(rbm.nv, rbm.nh)
    sampler_dw2000 = DW2000_32x8Sampler()
    sampler_dwadv = DWAdvantage_32x8Sampler()
    sampler_gibbs.set_weights(rbm.w, rbm.bv, rbm.bh)
    sampler_dw2000.set_weights(rbm.w, rbm.bv, rbm.bh)
    sampler_dwadv.set_weights(rbm.w, rbm.bv, rbm.bh)
    
    for n in N:
        vs_gibbs, _ = sampler_gibbs.get_samples(n)
        kl_gibbs = rbm.calc_kl_divergence(vs_gibbs, duplicate=True)
        vs_dw2000, _ = sampler_dw2000.get_samples(n)
        kl_dw2000 = rbm.calc_kl_divergence(vs_dw2000, duplicate=True)
        vs_dwadv, _ = sampler_dwadv.get_samples(n)
        kl_dwadv = rbm.calc_kl_divergence(vs_dwadv, duplicate=True)
        print(f'{kl_gibbs},{kl_dw2000},{kl_dwadv}')
    