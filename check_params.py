import numpy as np
import os
from rbm import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = SCRIPT_DIR + '/results/compare_train_with_calib/220608032950/model0.npz'
BETA0 = 3.11596654352799
BETA1 = 3.11596654352799
BETA2 = 3.11596654352799

if __name__ == '__main__':
    rbm = RBM(0, 0)
    rbm.load(MODEL_FILE)
    rbm.w /= BETA0
    rbm.bv /= BETA1
    rbm.bh /= BETA2

    j, h = rbm.get_ising()
    print(j)
    print(h)
    
