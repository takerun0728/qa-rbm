from rbm import *
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

LOAD_FILE = os.path.dirname(os.path.abspath(__file__)) + '/results/compare_cd/230812054948/min_kl_train_1.npz'
CALIB_PARAMS = {'lr': 0.00001, 'decay': 0.985}
#CALIB_PARAMS = {'lr':0.01, 'beta1':0.9, 'beta2':0.999, 'eps':1e-8, 'decay':1}
#EPOCH = 0
EPOCH = 200
BATCH_SAMPLE_NUM = 1012
#EVAL_SAMPLE_NUM = 101200#0
EVAL_SAMPLE_NUM = 1012000
#EVAL_SAMPLE_NUM = 92000
COMP_SAMPLE_NUMS = [1000, 10000, 100000, 1000000]
#COMP_SAMPLE_NUMS = [1000, 10000]
MANY_N_TH = 0
#MANY_N_TH = 5
MANY_N = 10
#MODE = 'DWAVEAdvantage'
MODE = 'GIBBS'
K = 7
#SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#SEED = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#SEED = [31, 32, 33, 34, 35, 36, 37, 38, 39, 30]
#SEED = [41, 42, 43, 44, 45, 46, 47, 48, 49, 40]
#SEED = [51, 52, 53, 54, 55, 56, 57, 58, 59, 50]
#SEED = [61, 62, 63, 64, 65, 66, 67, 68, 69, 60]
#SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
SEED = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 30]
def generate_hist(energies, ideal_energies, filename, binsize):
    plt.cla()
    l = int(np.floor(np.min([np.min(energies), np.min(ideal_energies)])))
    h = int(np.ceil(np.max([np.max(energies), np.max(ideal_energies)])))
    plt.hist(ideal_energies, alpha=0.3, color='orange', bins=np.arange(l, h, binsize))
    plt.hist(energies, alpha=0.3, color='blue', bins=np.arange(l, h, binsize))
    plt.savefig(filename)

def calc_kl_average(rbm, vs, hs, n):
    tmp_kls = []
    for i in range(len(vs) // n):
        tmp_kls.append(rbm.calc_kl_divergence_all_visible(vs[n*i:n*(i+1), :], hs[n*i:n*(i+1), :], duplicate=True))
    return np.average(tmp_kls)

if __name__ == '__main__':
    np.random.seed(0)
    now = datetime.datetime.now()
    save_dir = os.path.dirname(os.path.abspath(__file__)) + now.strftime('/results/compare_calib/%y%m%d%H%M%S')
    os.makedirs(save_dir)

    conf_log = ''
    conf_log += f'Model file: {LOAD_FILE}\n'
    conf_log += f'Calibration parameters: {CALIB_PARAMS}\n'
    conf_log += f'Number of epochs: {EPOCH}\n'
    conf_log += f'Number of batch samples: {BATCH_SAMPLE_NUM}\n'
    conf_log += f'Number of evaluation samples {EVAL_SAMPLE_NUM}\n'
    conf_log += f'Numbers of compare samples {COMP_SAMPLE_NUMS}\n'
    conf_log += f'Threshold epoch number for chaging calibration n: {MANY_N_TH}\n'
    conf_log += f'First calibration n : {MANY_N}\n'
    conf_log += f'Mode: {MODE}\n'
    conf_log += f'Seed: {SEED}\n'
    conf_log += f'CD-K: {K}\n'

    tmp_rbm = RBM(0, 0)
    tmp_rbm.load(LOAD_FILE)

    if MODE == 'GIBBS':
        sampler = RBMSamplerWithAllBiasError(tmp_rbm.nv, tmp_rbm.nh, error_strength=1.0, error_offset_w=3.0, error_offset_bv=5.0, error_offset_bh=2.0)
        conf_log += f'Gibbs Error: w:{sampler.w_error}, bv:{sampler.bv_error}, bh:{sampler.bh_error}'
        print(f"Error: w:{sampler.w_error}, bv:{sampler.bv_error}, bh:{sampler.bh_error}")
    elif MODE == 'DWAVE2000':
        sampler = DW2000_32x8Sampler()
        pass
    elif MODE == 'DWAVEAdvantage':
        sampler = DWAdvantage_32x8Sampler()
        pass
    
    print(conf_log)
    f = open(save_dir + '/config_log.txt', 'w')
    f.write(conf_log)
    f.close()
    
    kls = np.zeros((len(SEED), len(COMP_SAMPLE_NUMS), 5))
    for k, s in enumerate(SEED):
        np.random.seed(s)
        os.makedirs(save_dir + f'/{s}')
        print(f'Seeed is {s}')

        rbm_one = RBMOneCoefCalibWithSampler(0, 0, calib_optimizer='sgd', calib_w_params=CALIB_PARAMS, calib_bv_params=CALIB_PARAMS, calib_bh_params=CALIB_PARAMS) #nv and nh are automatically set when loading
        rbm_each = RBMEachCoefCalibWithSampler(0, 0, calib_optimizer='sgd', calib_w_params=CALIB_PARAMS, calib_bv_params=CALIB_PARAMS, calib_bh_params=CALIB_PARAMS)
        rbm_allbias = RBMAllBiasCalibWithSampler(0, 0, calib_optimizer='sgd', calib_w_params=CALIB_PARAMS, calib_bv_params=CALIB_PARAMS, calib_bh_params=CALIB_PARAMS)
        rbms = [rbm_one, rbm_each, rbm_allbias]
        calib_process = [np.zeros((EPOCH, 4)), np.zeros((EPOCH, 4)), np.zeros((EPOCH, 2 + tmp_rbm.nv + tmp_rbm.nh))]

        print(f'Generating ideal samples')
        ideal_sampler = RBMSampler(tmp_rbm.nv, tmp_rbm.nh)
        ideal_sampler.set_weights(tmp_rbm.w, tmp_rbm.bv, tmp_rbm.bh)
        ideal_vs, ideal_hs = ideal_sampler.get_samples(EVAL_SAMPLE_NUM)
        a = calc_kl_average(tmp_rbm, ideal_vs, ideal_hs, 10000)
        
        for i, rbm in enumerate(rbms):
            print(f'Calibrating {i}')
            rbm.load(LOAD_FILE)
            if i == 1:
                rbm_each.alpha = rbm_one.alpha
                rbm_each.beta_v = rbm_one.alpha
                rbm_each.beta_h = rbm_one.alpha
            elif i == 2:
                rbm_allbias.alpha = rbm_each.alpha
                rbm_allbias.beta_v[:] = rbm_each.beta_v
                rbm_allbias.beta_h[:] = rbm_each.beta_h

            for j in range(EPOCH):
                calib_process[i][j, 0] = j
                calib_process[i][j, 1:] = np.hstack([rbm.alpha, rbm.beta_v, rbm.beta_h])
                print(f'{j} {np.average(rbm.alpha)} {np.average(rbm.beta_v)} {np.average(rbm.beta_h)}')
                sampler.set_weights(rbm.w / rbm.alpha, rbm.bv / rbm.beta_v, rbm.bh / rbm.beta_h)
                vs, hs = sampler.get_samples(BATCH_SAMPLE_NUM)
                if j <= MANY_N_TH:
                    rbm.calibrate(vs, hs, cd_k=K, n=MANY_N)
                else:
                    rbm.calibrate(vs, hs, cd_k=K, n=1)

            np.savetxt(save_dir + f'/{s}/calib_process{i}.csv', calib_process[i], delimiter=',')
            
            sampler.set_weights(rbm.w / rbm.alpha, rbm.bv / rbm.beta_v, rbm.bh / rbm.beta_h)
            eval_vs, eval_hs = sampler.get_samples(EVAL_SAMPLE_NUM)
            np.save(save_dir + f'/{s}/samples{i}', np.hstack([eval_vs, eval_hs]))
            
            for j, n in enumerate(COMP_SAMPLE_NUMS):
                if i == 0:
                    kls[k, j, 0] = n

                print(f"Calculating KL divergence {i} {n}")
                if i == 0:
                    kls[k, j, 1] =  calc_kl_average(tmp_rbm, ideal_vs[:n], ideal_hs[:n], n)
                kls[k, j, i + 2] =  calc_kl_average(tmp_rbm, eval_vs[:n], eval_hs[:n], n)

            print(f"Generating histgrams {i}")
            energies = tmp_rbm.get_energy(eval_vs, eval_hs)
            ideal_energies = tmp_rbm.get_energy(ideal_vs, ideal_hs)
            generate_hist(energies, ideal_energies, save_dir + f'/{s}/hist_total{i}.png', 0.5)

            energies = tmp_rbm.get_weight_energy(eval_vs, eval_hs)
            ideal_energies = tmp_rbm.get_weight_energy(ideal_vs, ideal_hs)
            generate_hist(energies, ideal_energies, save_dir + f'/{s}/hist_w{i}.png', 0.5)
            
            energies = tmp_rbm.get_visible_bias_energy(eval_vs)
            ideal_energies = tmp_rbm.get_visible_bias_energy(ideal_vs)
            generate_hist(energies, ideal_energies, save_dir + f'/{s}/hist_bv{i}.png', 0.5)
            
            energies = tmp_rbm.get_hidden_bias_energy(eval_hs)
            ideal_energies = tmp_rbm.get_hidden_bias_energy(ideal_hs)
            generate_hist(energies, ideal_energies, save_dir + f'/{s}/hist_bh{i}.png', 0.5)
        
        np.savetxt(save_dir + f'/{s}/kls.csv', kls[k], delimiter=',')
    print(kls[:,:,1:].mean(axis=0))
    print(kls[:,:,1:].std(axis=0))
    np.savetxt(save_dir + '/summary_kl.csv', np.hstack([kls[0,:,0].reshape(-1,1), kls[:,:,1:].mean(axis=0), kls[:,:,1:].std(axis=0)]), delimiter=',')