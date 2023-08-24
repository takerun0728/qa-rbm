from pickle import TRUE
from rbm import *
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

GPU = True  #Only used for calculating KL divergence, not training
OPTIMIZER = 'adam'
CALIB_PARAMS = {'lr': 0.0000001, 'decay': 1}
CALIB_PARAMS2 = {'lr': 0.0000005, 'decay': 1}
OPT_PARAMS = {'lr':0.01, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':1}
EPOCH = 2000
CALC_KL_STEP = 10
#EPOCH = 50
NUM_SAMPLES = 1200
#BATCH = 18078
BATCH = 1000
#EVAL_SAMPLE_NUM = 10000
#COMP_SAMPLE_NUMS = [1000, 10000]
MANY_N_TH = 0
#MANY_N_TH = 5200
MANY_N = 1000
MODE = 'DWAVEAdvantage'
#MODE = 'DWAVE2000'
#MODE = 'GIBBS'
#J_LIMIT = [-1, 1]
#H_LIMIT = [-2, 2]
#NUM_TRAIN_DATA = 18078
#NUM_TEST_DATA = 4923
NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 1000
NUM_HIDDEN = 8
SEED = 2
CALIB_N = 20
EPOCH_TH_ONE_PARAM = 500
EPOCH_TH_EACH_PARAM = 500
ONE_PARAM = False
EACH_COEF = False
ALL_BIAS = True

if __name__ == '__main__':
    np.random.seed(SEED)
    now = datetime.datetime.now()
    save_dir = os.path.dirname(os.path.abspath(__file__)) + now.strftime('/results/compare_train_with_calib/%y%m%d%H%M%S')
    os.makedirs(save_dir)

    conf_log = ''
    conf_log += f'Learn optimizer: {OPTIMIZER}\n'
    conf_log += f'Learn parameters: {OPT_PARAMS}\n'
    conf_log += f'Calibration parameters: {CALIB_PARAMS}\n'
    conf_log += f'Calibration parameters(for All Biases): {CALIB_PARAMS2}\n'
    conf_log += f'Number of epochs: {EPOCH}\n'
    conf_log += f'KL divergence calucation step: {CALC_KL_STEP}\n'
    conf_log += f'Number of learn samples: {NUM_SAMPLES}\n'
    conf_log += f'Number of hidden nodes: {NUM_HIDDEN}\n'
    conf_log += f'Calibration times: {CALIB_N}\n'
    conf_log += f'Batch size: {BATCH}\n'
    conf_log += f'Threshold epoch number for chaging calibration n: {MANY_N_TH}\n'
    conf_log += f'First calibration n : {MANY_N}\n'
    conf_log += f'Mode : {MODE}\n'
    #conf_log += f'Limitation of J: {J_LIMIT}\n'
    #conf_log += f'Limitation of H: {H_LIMIT}\n'
    conf_log += f'Epoch threshold of updating one parameter: {EPOCH_TH_ONE_PARAM}\n'
    conf_log += f'Epoch threshold of updating each parameter: {EPOCH_TH_EACH_PARAM}\n'
    conf_log += f'Seed: {SEED}\n'
    print(conf_log)

    f = open(save_dir + '/config_log.txt', 'w')
    f.write(conf_log)
    f.close()

    images = np.load(os.path.dirname(os.path.abspath(__file__)) + '/processed_mnist.npy')
    train_imgs = images[:60000, :]
    test_imgs = images[60000:70000, :]
    train_imgs = np.unique(train_imgs, axis=0) #There are duplicated data because of reducing dimension
    test_imgs = np.unique(test_imgs, axis=0)
    train_imgs = train_imgs[:NUM_TRAIN_DATA]
    test_imgs = test_imgs[:NUM_TEST_DATA]
    nv = images.shape[1]

    rbm_one = RBMOneCoefCalibWithSampler(nv, NUM_HIDDEN, optimizer=OPTIMIZER, opt_params=OPT_PARAMS, batch_size=BATCH, calib_w_params=CALIB_PARAMS, gpu=True)#, j_limit=J_LIMIT, h_limit=H_LIMIT) #nv and nh are automatically set when loading
    rbm_each = RBMEachCoefCalibWithSampler(nv, NUM_HIDDEN, optimizer=OPTIMIZER, opt_params=OPT_PARAMS, batch_size=BATCH, calib_w_params=CALIB_PARAMS, calib_bv_params=CALIB_PARAMS, calib_bh_params=CALIB_PARAMS, gpu=True)#, j_limit=J_LIMIT, h_limit=H_LIMIT)
    rbm_allbias = RBMAllBiasCalibWithSampler(nv, NUM_HIDDEN, optimizer=OPTIMIZER, opt_params=OPT_PARAMS, batch_size=BATCH, calib_w_params=CALIB_PARAMS, calib_bv_params=CALIB_PARAMS2, calib_bh_params=CALIB_PARAMS2, gpu=True)#, j_limit=J_LIMIT, h_limit=H_LIMIT)
    rbms = [rbm_one, rbm_each, rbm_allbias]
    calib_process = [np.zeros((EPOCH, 4)), np.zeros((EPOCH, 4)), np.zeros((EPOCH, 4))]
    kls = np.zeros((EPOCH // CALC_KL_STEP, 7))
    
    if MODE == 'GIBBS':
        sampler = RBMSamplerWithAllBiasError(nv, NUM_HIDDEN, error_strength=1.0, error_offset_w=7, error_offset_bv=6, error_offset_bh=5)
    elif MODE == 'DWAVE2000':
        sampler = DW2000_32x8Sampler()
    elif MODE == 'DWAVEAdvantage':
        sampler = DWAdvantage_32x8Sampler()

    for i, rbm in enumerate(rbms):
        if (not ONE_PARAM) and  (i == 0):
            continue
        if (not EACH_COEF) and  (i == 1):
            continue
        if (not ALL_BIAS) and  (i == 2):
            continue

        rbm.set_sampler(sampler)
        row = 0

        for epoch in range(EPOCH):
            if epoch <= MANY_N_TH:         
                rbm.train_sample(train_imgs, ns=NUM_SAMPLES, calib_n=MANY_N, calib_k=1)
            else:
                if i == 0:
                    rbm.train_sample(train_imgs, ns=NUM_SAMPLES, calib_n=CALIB_N, calib_k=1, mode='one')
                if i == 1:
                    if epoch < EPOCH_TH_ONE_PARAM:
                        rbm.train_sample(train_imgs, ns=NUM_SAMPLES, calib_n=CALIB_N, calib_k=1, mode='one')
                    else:
                        rbm.train_sample(train_imgs, ns=NUM_SAMPLES, calib_n=CALIB_N, calib_k=1, mode='each')
                elif i == 2:
                    if epoch < EPOCH_TH_ONE_PARAM:
                        rbm.train_sample(train_imgs, ns=NUM_SAMPLES, calib_n=CALIB_N, calib_k=1, mode='one')
                    elif epoch < EPOCH_TH_EACH_PARAM:
                        rbm.train_sample(train_imgs, ns=NUM_SAMPLES, calib_n=CALIB_N, calib_k=1, mode='each')
                    else:
                        rbm.train_sample(train_imgs, ns=NUM_SAMPLES, calib_n=CALIB_N, calib_k=1, mode='all')

            if epoch % CALC_KL_STEP == 0:
                kl_train = rbm.calc_kl_divergence(train_imgs)
                kl_test = rbm.calc_kl_divergence(test_imgs)
                print(f"Epoch {epoch} train:{kl_train} test:{kl_test}")
                print(f'Calib: {np.average(rbm.alpha)} {np.average(rbm.beta_v)} {np.average(rbm.beta_h)}')
                if i == 0: kls[row, 0] = epoch
                kls[row, i * 2 + 1] = kl_train
                kls[row, i * 2 + 2] = kl_test
                row += 1

            calib_process[i][epoch, 0] = epoch
            calib_process[i][epoch, 1:] = [np.average(rbm.alpha), np.average(rbm.beta_v), np.average(rbm.beta_h)]

        np.savetxt(save_dir + f'/calib_process{i}.csv', calib_process[i], delimiter=',')
        rbm.save(save_dir + f'/model{i}')
    np.savetxt(save_dir + f'/kls.csv', kls, delimiter=',')
