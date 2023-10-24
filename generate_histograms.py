from rbm import *
import numpy as np
import matplotlib.pyplot as plt

LOAD_MODEL = os.path.dirname(os.path.abspath(__file__)) + '/results/compare_cd/230825061647/min_kl_train_1.npz'
LOAD_SAMPLE = os.path.dirname(os.path.abspath(__file__)) + '/results/compare_calib/231024082408/1/'
L = [-45, -50, -15, -12]
H = [-10, 3, 0, 15]

BINSIZE = 0.5

if __name__ == '__main__':
    rbm = RBM(0, 0)
    rbm.load(LOAD_MODEL)
    ideal_energies = np.load(LOAD_SAMPLE + 'ideal_energies.npy')
    energies = []
    energies.append(np.load(LOAD_SAMPLE + 'energies0.npy'))
    energies.append(np.load(LOAD_SAMPLE + 'energies1.npy'))
    energies.append(np.load(LOAD_SAMPLE + 'energies2.npy'))

    fig = plt.figure()
    for i, energy in enumerate(energies):
        for j, divided in enumerate(energy):
            ax = fig.add_subplot(3, 4, i*4 + j + 1)
            ax.hist(ideal_energies[j], alpha=0.3, color='orange', bins=np.arange(L[j], H[j], BINSIZE))
            ax.hist(divided, alpha=0.3, color='blue', bins=np.arange(L[j], H[j], BINSIZE))
            if j != 0:
                ax.tick_params(labelleft=False)
            if i != 2:
                ax.tick_params(labelbottom=False)
            print(f'min {np.min(divided)}, max {np.max(divided)}')
    
    plt.show()
