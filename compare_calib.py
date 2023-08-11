from rbm import *
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

LOAD_FILE = os.path.dirname(os.path.abspath(__file__)) + '/results/220525233454/cd/min_kl_train_8.npz'
CALIB_PARAMS = {'lr': 0.0000005, 'decay': 1}
EPOCH = 500
#EPOCH = 50
BATCH_SAMPLE_NUM = 1200
EVAL_SAMPLE_NUM = 1200000
#EVAL_SAMPLE_NUM = 10000
COMP_SAMPLE_NUMS = [1000, 10000, 100000, 1000000]
#COMP_SAMPLE_NUMS = [1000, 10000]
MANY_N_TH = 50
#MANY_N_TH = 5
MANY_N = 1000
MODE = 'DWAVEAdvantage'
#MODE = 'GIBBS'
SEED = 9

def generate_hist(energies, ideal_energies, filename, binsize):
    plt.cla()
    l = int(np.floor(np.min([np.min(energies), np.min(ideal_energies)])))
    h = int(np.ceil(np.max([np.max(energies), np.max(ideal_energies)])))
    plt.hist(ideal_energies, alpha=0.3, color='orange', bins=range(l, h, binsize))
    plt.hist(energies, alpha=0.3, color='blue', bins=range(l, h, binsize))
    plt.savefig(filename)

def calc_kl_average(rbm, vs, hs, n):
    tmp_kls = []
    for i in range(len(vs) // n):
        tmp_kls.append(rbm.calc_kl_divergence_all_visible(vs[n*i:n*(i+1), :], hs[n*i:n*(i+1), :], duplicate=True))
    return np.average(tmp_kls)

if __name__ == '__main__':
    np.random.seed(SEED)
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
    conf_log += f'Mode : {MODE}\n'
    conf_log += f'Seed {SEED}\n'
    print(conf_log)

    f = open(save_dir + '/config_log.txt', 'w')
    f.write(conf_log)
    f.close()
    
    rbm_one = RBMOneCoefCalibWithSampler(0, 0, calib_w_params=CALIB_PARAMS) #nv and nh are automatically set when loading
    rbm_each = RBMEachCoefCalibWithSampler(0, 0, calib_w_params=CALIB_PARAMS, calib_bv_params=CALIB_PARAMS, calib_bh_params=CALIB_PARAMS)
    rbm_allbias = RBMAllBiasCalibWithSampler(0, 0, calib_w_params=CALIB_PARAMS, calib_bv_params=CALIB_PARAMS, calib_bh_params=CALIB_PARAMS)
    rbms = [rbm_one, rbm_each, rbm_allbias]
    calib_process = [np.zeros((EPOCH, 4)), np.zeros((EPOCH, 4)), np.zeros((EPOCH, 4))]
    kls = np.zeros((len(COMP_SAMPLE_NUMS), 5))

    print(f'Generating ideal samples')
    tmp_rbm = RBM(0, 0)
    tmp_rbm.load(LOAD_FILE)
    ideal_sampler = RBMSampler(tmp_rbm.nv, tmp_rbm.nh)
    ideal_sampler.set_weights(tmp_rbm.w, tmp_rbm.bv, tmp_rbm.bh)
    ideal_vs, ideal_hs = ideal_sampler.get_samples(EVAL_SAMPLE_NUM)
    
    if MODE == 'GIBBS':
        sampler = RBMSamplerWithEachCoefError(tmp_rbm.nv, tmp_rbm.nh, error_strength=0.1, error_offset=3)
    elif MODE == 'DWAVE2000':
        sampler = DW2000_32x8Sampler()
        pass
    elif MODE == 'DWAVEAdvantage':
        sampler = DWAdvantage_32x8Sampler()
        pass

    for i, rbm in enumerate(rbms):
        print(f'Calibrating {i}')
        rbm.load(LOAD_FILE)
        if i == 2:
            rbm_allbias.alpha = rbm_each.alpha
            rbm_allbias.beta_v[:] = rbm_each.beta_v
            rbm_allbias.beta_h[:] = rbm_each.beta_h

        for j in range(EPOCH):
            calib_process[i][j, 0] = j
            calib_process[i][j, 1:] = [np.average(rbm.alpha), np.average(rbm.beta_v), np.average(rbm.beta_h)]
            print(f'{j} {np.average(rbm.alpha)} {np.average(rbm.beta_v)} {np.average(rbm.beta_h)}')
            sampler.set_weights(rbm.w / rbm.alpha, rbm.bv / rbm.beta_v, rbm.bh / rbm.beta_h)
            vs, hs = sampler.get_samples(BATCH_SAMPLE_NUM)
            if j <= MANY_N_TH:
                rbm.calibrate(vs, hs, cd_k=1, n=MANY_N)
            else:
                rbm.calibrate(vs, hs, cd_k=1, n=1)

        np.savetxt(save_dir + f'/calib_process{i}.csv', calib_process[i], delimiter=',')
        sampler.set_weights(rbm.w / rbm.alpha, rbm.bv / rbm.beta_v, rbm.bh / rbm.beta_h)
        eval_vs, eval_hs = sampler.get_samples(EVAL_SAMPLE_NUM)
        np.save(save_dir + f'/samples{i}', np.hstack([eval_vs, eval_hs]))
        
        for j, n in enumerate(COMP_SAMPLE_NUMS):
            if i == 0:
                kls[j, 0] = n

            print(f"Calculating KL divergence {i} {n}")
            if i == 0:
                kls[j, 1] =  calc_kl_average(tmp_rbm, ideal_vs, ideal_hs, n)
            kls[j, i + 2] =  calc_kl_average(tmp_rbm, eval_vs, eval_hs, n)

        print(f"Generating histgrams {i}")
        energies = tmp_rbm.get_energy(eval_vs, eval_hs)
        ideal_energies = tmp_rbm.get_energy(ideal_vs, ideal_hs)
        generate_hist(energies, ideal_energies, save_dir + f'/hist_total{i}.png', 1)

        energies = tmp_rbm.get_weight_energy(eval_vs, eval_hs)
        ideal_energies = tmp_rbm.get_weight_energy(ideal_vs, ideal_hs)
        generate_hist(energies, ideal_energies, save_dir + f'/hist_w{i}.png', 1)
        
        energies = tmp_rbm.get_visible_bias_energy(eval_vs)
        ideal_energies = tmp_rbm.get_visible_bias_energy(ideal_vs)
        generate_hist(energies, ideal_energies, save_dir + f'/hist_bv{i}.png', 1)
        
        energies = tmp_rbm.get_hidden_bias_energy(eval_hs)
        ideal_energies = tmp_rbm.get_hidden_bias_energy(ideal_hs)
        generate_hist(energies, ideal_energies, save_dir + f'/hist_bh{i}.png', 2)
    
    np.savetxt(save_dir + f'/kls.csv', kls, delimiter=',')