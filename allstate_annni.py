import numpy as np
from transfer_matrix_lib import *

L = 4
J = 1
K = 1/2
H = 0
BETA = 1

if __name__ == '__main__':
    xs = [[i, j, k, l] for i in range(2**L) for j in range(2**L) for k in range(2**L) for l in range(2**L)]
    es = []
    for x in xs:
        e = 0
        for r in x:
            e += vx(r, L, J, K, H)
        e += vz(x[-1], x[0], L, J)
        e += vz(x[0], x[1], L, J)
        e += vz(x[1], x[2], L, J)
        e += vz(x[2], x[3], L, J)

        es.append(e)
    es = np.array(es)
    print(np.sum(np.exp(-BETA * es)))