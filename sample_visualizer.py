import os
import numpy as np
import matplotlib.pyplot as plt

LOAD_FILE = os.path.dirname(os.path.abspath(__file__)) + '/results/compare_calib/230812222348/3/samples2.npy'
N = 50
COL = 10
ROW = N // COL

if __name__ == '__main__':
    samples = np.load(LOAD_FILE)
    samples = samples[:, :32]
    samples = np.insert(samples, 0, np.zeros(samples.shape[0]), axis=1)
    samples = np.insert(samples, 5, np.zeros(samples.shape[0]), axis=1)
    samples = np.insert(samples, 30, np.zeros(samples.shape[0]), axis=1)
    samples = np.insert(samples, 35, np.zeros(samples.shape[0]), axis=1)
    samples = samples.reshape(-1, 6, 6)

    #Display samples of training data
    fig = plt.figure()
    for i, sample in enumerate(samples[0:N]):
        ax = fig.add_subplot(ROW, COL, i + 1)
        ax.imshow(sample)
    plt.show()