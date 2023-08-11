import numpy as np
import matplotlib.pyplot as plt
from rbm import * 

if __name__ == '__main__':
    np.random.seed(0)

    N = 20
    side = 2
    nv = side**2
    nh = 16
    images = np.round(np.random.rand(N, nv))
    
    # train
    rbm = RBM(nv, nh, batch_size=4)
    #rbm.train_pcd(images, t=1, epoch=200, lr=0.1)
    for epoch in range(20000):
        if epoch % 10 == 0:
            #cost = rbm.calc_mpf_cost(images)
            #print(f"{epoch}:{cost}")
            print(f"{epoch}:{rbm.calc_kl_divergence(images, True)}")
        #rbm.train_mpf(images)
        #rbm.train_sa(images, mcmc_iter=100, burn_in=50)
        rbm.train_cd(images, t=16)
    pass

    fig = plt.figure()
    for i in range(10):
        ax = fig.add_subplot(2, 10, i + 1)
        ax.imshow(images[i].reshape(side, side))
        ax = fig.add_subplot(2, 10, i + 11)
        h, ph = rbm.get_hidden(images[i])
        v, pv = rbm.get_visible(h)
        ax.imshow(pv.reshape(side, side))
    plt.show()


    
