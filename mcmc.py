import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

N = 100
COL = 128
ROW = 128
C = 2
J = 1
H = 0
B = np.array([2.0, 1.1, 1.0, 0.9, 0.5]) / 2.26

def e(x):
    return -(J * (np.sum(x[:-1, :] * x[1:, :]) + np.sum(x[:, :-1] * x[:, 1:])) + H * np.sum(x))

def dE(x, i, j):
    de = x[i - 1, j] if i != 0 else 0
    de += x[i + 1, j] if i != ROW - 1 else 0
    de += x[i, j - 1] if j != 0 else 0
    de += x[i, j + 1] if j != COL - 1 else 0
    return 2 * (J * de + H) * x[i, j]

samples = []
energies = []
update_cnt = 0

for b in B:
    x = np.random.randint(0, 2, size=(ROW, COL)) * 2 - 1
    samples.append([])
    energies.append([])
    for n in range(N):
        for i in range(ROW):
            for j in range(COL):
                if rng.random() < np.exp(-b * dE(x, i, j)):
                    x[i, j] *= -1
        samples[-1].append(x)
        energies[-1].append(e(x))

#energies = np.array([e(x) for x in samples])
samples = np.array(samples)
energies = np.array(energies)

fig = plt.figure()
for i, (sample, energy) in enumerate(zip(samples[:, 0, :, :], energies[:, :])):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(sample)
    ax = fig.add_subplot(2, 5, i + 6)
    ax.plot(energy)
plt.show()

