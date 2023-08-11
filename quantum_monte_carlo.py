import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

M = 32
N = 100
COL = 128
ROW = 128
C = 2
J = 1
H = 0
B = np.array([2.0, 1.1, 1.0, 0.9, 0.5]) / 2.26

def e(x):
    return -(J * (np.sum(x[:-1, :] * x[1:, :]) + np.sum(x[:, :-1] * x[:, 1:])) + H * np.sum(x))

def dE(x, i, m, j, sigma, gamma, beta):
    de = (np.sum(x[m] * j) - x[m, i] * j[i]) / M
    de += sigma[i] / M
    tmp = x[m - 1, i]
    tmp += x[m + 1, i] if m != M - 1 else x[0, i]
    de += -np.log(np.tanh(gamma * beta / m)) / (2 * beta) * tmp
    
    return 2 * de * x[i]

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
