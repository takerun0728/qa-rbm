from rbm import *
import numpy as np
import matplotlib.pyplot as plt

LOAD_MODEL = os.path.dirname(os.path.abspath(__file__)) + '/results/compare_cd/230825061647/min_kl_train_1.npz'
LOAD_SAMPLE = os.path.dirname(os.path.abspath(__file__)) + '/results/compare_calib/231024052257/1/'
L = -10
H = 50
BINSIZE = 0.5

if __name__ == '__main__':
    rbm = RBM(0, 0)
    rbm.load(LOAD_MODEL)
    ideal_energies = np.load(LOAD_SAMPLE + 'ideal_energies.npy')
    energies = []
    energies.append(np.load(LOAD_SAMPLE + 'energies0.npy'))
    energies.append(np.load(LOAD_SAMPLE + 'energies1.npy'))
    energies.append(np.load(LOAD_SAMPLE + 'energies2.npy'))

    fig = plt.figure(figsize=(3, 4))
    for i, energy in enumerate(energies):
        for j, divied in enumerate(energy):
            ax = fig.add_subplot(i, j)
            ax.hist()
            ax.hist(ideal_energies[j], alpha=0.3, color='orange', bins=np.arange(L, H, BINSIZE))
            ax.hist(divied, alpha=0.3, color='blue', bins=np.arange(L, H, BINSIZE))
    
    plt.show()
