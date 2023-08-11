from pickle import TRUE
import matplotlib.pyplot as plt
from rbm import * 
import os
import datetime
import numpy as np

GPU = True  #Only used for calculating KL divergence, not training
CD = True
MPF = False
GIBBS = True
OPTIMIZER = 'adam'
OPT_PARAMS_CD = {'lr':0.01, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':1}
OPT_PARAMS_MPF = {'lr':0.01, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':1}
OPT_PARAMS_GIBBS = {'lr':0.01, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':1}
GIBBS_ITER = 1010
GIBBS_BURN_IN = 10
CD_K = 128
EPOCH_CD = 2000
EPOCH_MPF = 1500
EPOCH_GIBBS = 2000
BATCH = 1000
HIDDENS = [8]
SEED = 0
CALC_KL_STEP = 10
HEADER = 'CD, CD-k, MPF, Gibbs'
NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 1000
J_LIMIT = [-np.inf, np.inf]
H_LIMIT = [-np.inf, np.inf]

def generate_conf_log(dir):
    conf_log = ''
    conf_log += f'Use GPU: {GPU}\n'
    conf_log += f'Learn by CD: {CD}\n'
    conf_log += f'Learn by MPF: {MPF}\n'
    conf_log += f'Learn by Gibbs: {GIBBS}\n'
    conf_log += f'Optimizer: {OPTIMIZER}\n'
    conf_log += f'Optimize parameters for CD: {OPT_PARAMS_CD.__str__()}\n'
    conf_log += f'Optimize parameters for MPF: {OPT_PARAMS_MPF.__str__()}\n'
    conf_log += f'Optimize parameters for Gibbs: {OPT_PARAMS_GIBBS.__str__()}\n'
    conf_log += f'Number of gibbs iterations: {GIBBS_ITER}\n'
    conf_log += f'Number of gibbs burn-in: {GIBBS_BURN_IN}\n'
    conf_log += f'Nuber of EPOCH for CD: {EPOCH_CD}\n'
    conf_log += f'Nuber of EPOCH for MPF: {EPOCH_MPF}\n'
    conf_log += f'Nuber of EPOCH for Gibbs: {EPOCH_GIBBS}\n'
    conf_log += f'Mini batch size: {BATCH}\n'
    conf_log += f'Nubers list of Hidden nodes: {HIDDENS}\n'
    conf_log += f'Seed {SEED}\n'
    conf_log += f'Step of calculating KL divergence {CALC_KL_STEP}\n'
    print(conf_log)

    f = open(dir + '/config_log.txt', 'w')
    f.write(conf_log)
    f.close()

def train(nv, nh, train_mode, dir):
    kl_results = []
    min_kl_train = np.inf
    min_kl_test = np.inf

    if train_mode == 'cd':
        epoch = EPOCH_CD
        rbm = RBM(nv, nh, optimizer=OPTIMIZER, opt_params=OPT_PARAMS_CD, batch_size=BATCH, gpu=GPU, j_limit=J_LIMIT, h_limit=H_LIMIT)
    if train_mode == 'cd_k':
        epoch = EPOCH_CD
        rbm = RBM(nv, nh, optimizer=OPTIMIZER, opt_params=OPT_PARAMS_CD, batch_size=BATCH, gpu=GPU, j_limit=J_LIMIT, h_limit=H_LIMIT)
    elif train_mode == 'mpf':
        epoch = EPOCH_MPF
        rbm = RBM(nv, nh, optimizer=OPTIMIZER, opt_params=OPT_PARAMS_MPF, batch_size=BATCH, gpu=GPU, j_limit=J_LIMIT, h_limit=H_LIMIT)
    elif train_mode == 'gibbs':
        epoch = EPOCH_GIBBS
        rbm = RBM(nv, nh, optimizer=OPTIMIZER, opt_params=OPT_PARAMS_GIBBS, batch_size=BATCH, gpu=GPU, j_limit=J_LIMIT, h_limit=H_LIMIT)

    for epoch in range(epoch):
        if epoch % CALC_KL_STEP == 0:
            kl_train = rbm.calc_kl_divergence(train_imgs)
            kl_test = rbm.calc_kl_divergence(test_imgs)
            print(f"Epoch {epoch} train:{kl_train} test:{kl_test}")
            if kl_train < min_kl_train:
                min_kl_train = kl_train
                rbm.save(dir + f'/{train_mode}/min_kl_train_{nh}')
            if kl_test < min_kl_test:
                min_kl_test = kl_test
                rbm.save(dir + f'/{train_mode}/min_kl_test_{nh}')
            kl_results.append([kl_train, kl_test])

        if train_mode == 'cd':
            rbm.train_cd(train_imgs, 1)
        if train_mode == 'cd_k':
            rbm.train_cd(train_imgs, CD_K)
        elif train_mode == 'mpf':
            rbm.train_mpf(train_imgs)
        elif train_mode == 'gibbs':
            rbm.train_gibbs(train_imgs, GIBBS_ITER, GIBBS_BURN_IN)

    return np.array(kl_results), min_kl_train, min_kl_test

if __name__ == '__main__':
    now = datetime.datetime.now()
    save_dir = os.path.dirname(os.path.abspath(__file__)) + now.strftime('/results/compare_bm/%y%m%d%H%M%S')
    os.makedirs(save_dir)
    if CD: os.makedirs(save_dir + '/cd')
    if CD: os.makedirs(save_dir + '/cd_k')
    if MPF: os.makedirs(save_dir + '/mpf')
    if GIBBS: os.makedirs(save_dir + '/gibbs')
    generate_conf_log(save_dir)

    np.random.seed(SEED)

    images = np.load(os.path.dirname(os.path.abspath(__file__)) + '/processed_mnist.npy')
    train_imgs = images[:60000, :]
    test_imgs = images[60000:70000, :]
    train_imgs = np.unique(train_imgs, axis=0) #There are duplicated data because of reducing dimension
    test_imgs = np.unique(test_imgs, axis=0)
    train_imgs = train_imgs[:NUM_TRAIN_DATA]
    test_imgs = test_imgs[:NUM_TEST_DATA]
    nv = images.shape[1]

    min_kl_train_list = np.ones((len(HIDDENS), 4)) * -1
    min_kl_test_list = np.ones((len(HIDDENS), 4)) * -1

    for i, nh in enumerate(HIDDENS):
        if CD:
            print(f'\n----------------------------------------------------')
            print(f'Start training with CD-1, size of hidden nodes is {nh}')
            kl_results, min_kl_train_list[i, 0], min_kl_test_list[i, 0] = train(nv, nh, 'cd', save_dir)
            np.savetxt(save_dir + f'/cd/{nh}.csv', kl_results, delimiter=',', header='train, test')
            np.savetxt(save_dir + f'/min_kl_train.csv', min_kl_train_list, delimiter=',', header=HEADER)
            np.savetxt(save_dir + f'/min_kl_test.csv', min_kl_test_list, delimiter=',', header=HEADER)

            print(f'\n----------------------------------------------------')
            print(f'Start training with CD-k, size of hidden nodes is {nh}')
            kl_results, min_kl_train_list[i, 1], min_kl_test_list[i, 1] = train(nv, nh, 'cd_k', save_dir)
            np.savetxt(save_dir + f'/cd_k/{nh}.csv', kl_results, delimiter=',', header='train, test')
            np.savetxt(save_dir + f'/min_kl_train.csv', min_kl_train_list, delimiter=',', header=HEADER)
            np.savetxt(save_dir + f'/min_kl_test.csv', min_kl_test_list, delimiter=',', header=HEADER)

        if MPF:
            print(f'\n----------------------------------------------------')
            print(f'Start training with MPF, size of hidden nodes is {nh}')
            kl_results, min_kl_train_list[i, 2], min_kl_test_list[i, 2] = train(nv, nh, 'mpf', save_dir)
            np.savetxt(save_dir + f'/mpf/{nh}.csv', kl_results, delimiter=',', header='train, test')
            np.savetxt(save_dir + f'/min_kl_train.csv', min_kl_train_list, delimiter=',', header=HEADER)
            np.savetxt(save_dir + f'/min_kl_test.csv', min_kl_test_list, delimiter=',', header=HEADER)

        if GIBBS:
            print(f'\n----------------------------------------------------')
            print(f'Start training with Gibbs, size of hidden nodes is {nh}')
            kl_results, min_kl_train_list[i, 3], min_kl_test_list[i, 3] = train(nv, nh, 'gibbs', save_dir)
            np.savetxt(save_dir + f'/gibbs/{nh}.csv', kl_results, delimiter=',', header='train, test')
            np.savetxt(save_dir + f'/min_kl_train.csv', min_kl_train_list, delimiter=',', header=HEADER)
            np.savetxt(save_dir + f'/min_kl_test.csv', min_kl_test_list, delimiter=',', header=HEADER)
