from pickle import TRUE
import matplotlib.pyplot as plt
from rbm import * 
import os
import datetime
import numpy as np

GPU = True  #Only used for calculating KL divergence, not training
OPTIMIZER = 'adam'
#OPT_PARAMS = {'lr':0.005, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':0.999997}
OPT_PARAMS = {'lr':0.01, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':1}
EPOCH = 1000
BATCH = 1000
HIDDEN = 8
SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CALC_KL_STEP = 20
GIBBS_ITER = 1010
GIBBS_BURN_IN = 10

def generate_conf_log(dir):
    conf_log = ''
    conf_log += f'Use GPU: {GPU}\n'
    conf_log += f'Optimizer: {OPTIMIZER}\n'
    conf_log += f'Optimize parameters: {OPT_PARAMS.__str__()}\n'
    conf_log += f'Nuber of EPOCH: {EPOCH}\n'
    conf_log += f'Mini batch size: {BATCH}\n'
    conf_log += f'Nubers of Hidden node: {HIDDEN}\n'
    conf_log += f'Burn in {GIBBS_BURN_IN}\n'
    conf_log += f'Number of samples: {GIBBS_ITER}\n'
    conf_log += f'Seed {SEED}\n'
    conf_log += f'Step of calculating KL divergence {CALC_KL_STEP}\n'
    print(conf_log)

    f = open(dir + '/config_log.txt', 'w')
    f.write(conf_log)
    f.close()

def train(nv, nh, dir):
    kl_results = []
    min_kl_train = np.inf
    min_kl_test = np.inf
    rbm = RBM(nv, nh, optimizer=OPTIMIZER, opt_params=OPT_PARAMS, batch_size=BATCH, gpu=GPU)

    for epoch in range(EPOCH):
        if epoch % CALC_KL_STEP == 0:
            kl_train = rbm.calc_kl_divergence(train_imgs)
            kl_test = rbm.calc_kl_divergence(test_imgs)
            print(f"Epoch {epoch} train:{kl_train} test:{kl_test}")
            if kl_train < min_kl_train:
                min_kl_train = kl_train
                rbm.save(dir + f'/min_kl_train_{s}')
            if kl_test < min_kl_test:
                min_kl_test = kl_test
                rbm.save(dir + f'/min_kl_test_{s}')
            kl_results.append([kl_train, kl_test])
        
        rbm.train_gibbs(train_imgs, GIBBS_ITER, GIBBS_BURN_IN)

    return np.array(kl_results), min_kl_train, min_kl_test

if __name__ == '__main__':
    now = datetime.datetime.now()
    save_dir = os.path.dirname(os.path.abspath(__file__)) + now.strftime('/results/compare_gibbs/%y%m%d%H%M%S')
    os.makedirs(save_dir)
    generate_conf_log(save_dir)

    np.random.seed(SEED[0])

    images = np.load(os.path.dirname(os.path.abspath(__file__)) + '/processed_mnist.npy')
    train_imgs = images[:60000, :]
    test_imgs = images[60000:70000, :]
    train_imgs = np.unique(train_imgs, axis=0) #There are duplicated data because of reducing dimension
    test_imgs = np.unique(test_imgs, axis=0)
    
    nv = images.shape[1]

    min_kl_train_list = np.ones((1, len(SEED)))
    min_kl_test_list = np.ones((1, len(SEED)))

    for j, s in enumerate(SEED):
        np.random.seed(s)
        print(f'\n----------------------------------------------------')
        print(f'Start training with Gibbs, seed is {s}')
        kl_results, min_kl_train_list[0, j], min_kl_test_list[0, j] = train(nv, HIDDEN, save_dir)
        np.savetxt(save_dir + f'/{s}.csv', kl_results, delimiter=',', header='train, test')
    np.savetxt(save_dir + f'/min_kl_train.csv', min_kl_train_list, delimiter=',')
    np.savetxt(save_dir + f'/min_kl_test.csv', min_kl_test_list, delimiter=',')
