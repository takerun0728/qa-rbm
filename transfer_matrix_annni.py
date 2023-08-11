import numpy as np
from transfer_matrix_lib import *

L = 4
J = 1
K = 1/2
H = 0
BETA = 1

if __name__ == '__main__':
    Tx = np.array([[np.exp(-BETA * vx(i, L, J, K, H)) if i == j else 0 for i in range(2**L)] for j in range(2**L)])
    Tz = np.array([[np.exp(-BETA * vz(i, j, L, J)) for i in range(2**L)] for j in range(2**L)])

    Tx_sqrt = np.sqrt(Tx)
    T = Tx_sqrt @ Tz @ Tx_sqrt
    T_tmp = Tx @ Tz
    w = np.linalg.eigvalsh(T)
    print((w**L).sum())
    pass