from dwave.system import DWaveSampler, FixedEmbeddingComposite
import numpy as np
import cupy as cp
from dimod.utilities import qubo_to_ising
from itertools import product
import cxxjij
import os

class SGD:
    def __init__(self, opt_params):
        self.lr = opt_params['lr']
        self.decay = opt_params['decay']
    
    def __call__(self, grad):
        ret = self.lr * grad
        self.lr *= self.decay
        return ret

class Momentum:
    def __init__(self, opt_params):
        self.lr = opt_params['lr']
        self.decay = opt_params['decay']
        self.alpha = opt_params['alpha']
        self.delta = 0

    def __call__(self, grad):
        self.delta = (1 - self.alpha) * self.lr * grad + self.alpha * self.delta
        self.lr *= self.decay
        return self.delta

class AdaGrad:
    def __init__(self, opt_params):
        self.lr = opt_params['lr']
        self.decay = opt_params['decay']
        self.h = 0
    
    def __call__(self, grad):
        self.h = self.h + grad**2
        ret = self.lr * grad / np.sqrt(self.h)
        self.lr *= self.decay
        return ret

class RMSProp:
    def __init__(self, opt_params):
        self.lr = opt_params['lr']
        self.rho = opt_params['rho']
        self.decay = opt_params['decay']
        self.h = 0
    
    def __call__(self, grad):
        self.h = self.rho * self.h + (1 - self.rho) * grad**2 + 0.00001
        ret = self.lr * grad / np.sqrt(self.h)
        self.lr *= self.decay
        return ret

class Adam:
    def __init__(self, opt_params):
        self.lr = opt_params['lr']
        self.beta1 = opt_params['beta1']
        self.beta2 = opt_params['beta2']
        self.eps = opt_params['eps']
        self.decay = opt_params['decay']
        self.m = 0
        self.v = 0
    
    def __call__(self, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        ret = self.lr * self.m / (np.sqrt(self.v) + self.eps)
        self.lr *= self.decay
        return ret

def sigmoid(z, xp=np):
    return 1 / (1 + xp.exp(-z))

def logsumexp(x, xp=np):
    max_x = xp.max(x) / 4
    return max_x + xp.log(xp.sum(xp.exp(x - max_x)))

def select_optimizer(optimizer, opt_params):
    if optimizer == 'sgd':
        return SGD(opt_params)
    elif optimizer == 'momentum':
        return Momentum(opt_params)
    elif optimizer == 'adagrad':
        return AdaGrad(opt_params)
    elif optimizer == 'rmsprop':
        return RMSProp(opt_params)
    elif optimizer == 'adam':
        return Adam(opt_params)
    else:
        return None

class RBM:
    CALC_PARTITION_DIV_DIM = 12

    def __init__(self, nv, nh, optimizer='momentum', opt_params={'lr':0.01, 'alpha':0.9, 'decay':1}, batch_size=100, gpu=False, j_limit=None, h_limit=None):
        self.nv = nv
        self.nh = nh
        self.bv = np.array(np.random.rand(nv)) * 0
        self.bh = np.array(np.random.rand(nh)) * 0
        self.w = np.array(np.random.rand(nv, nh)) * 0
        self.gpu = gpu
        self.batch_size = batch_size
        self.j_limit = j_limit
        self.h_limit = h_limit

        self.optimizer_w = select_optimizer(optimizer, opt_params)
        self.optimizer_bv = select_optimizer(optimizer, opt_params)
        self.optimizer_bh = select_optimizer(optimizer, opt_params)

    def calc_log_partition_func(self):
        xp = cp if self.gpu else np
        w = xp.asarray(self.w)
        bv = xp.asarray(self.bv)
        bh = xp.asarray(self.bh)
        
        if self.nh > self.CALC_PARTITION_DIV_DIM:
            nh_l = self.CALC_PARTITION_DIV_DIM
            nh_h = self.nh - self.CALC_PARTITION_DIV_DIM
            hs_l = xp.array(list(product([0, 1], repeat=nh_l)))
            hs_h = xp.array(list(product([0, 1], repeat=nh_h)))
            log_prob_vvs = xp.empty(2**nh_h)


            for i, h_h in enumerate(hs_h):
                h_h_w = h_h @ w[:, nh_l:].T
                hs_l_w = hs_l @ w[:, :nh_l].T
                hs_w = hs_l_w + h_h_w
                h_h_b = h_h @ bh[nh_l:].reshape(-1, 1)
                hs_l_b = hs_l @ bh[:nh_l].reshape(-1, 1)
                hs_b = (hs_l_b + h_h_b).ravel()
                log_prob_vv = hs_b + xp.sum(xp.log(1 + xp.exp(bv + hs_w)), axis=1)
                log_prob_vvs[i] = logsumexp(log_prob_vv, xp)
            return logsumexp(log_prob_vvs, xp)

        else:
            hs = xp.array(list(product([0, 1], repeat=self.nh)))
            log_prob = (hs @ bh.reshape(-1, 1)).ravel() + xp.sum(xp.log(1 + xp.exp(bv + (hs @ w.T))), axis=1)
            return logsumexp(log_prob, xp)

    def calc_kl_divergence(self, data, duplicate=False):
        log_z = self.calc_log_partition_func()
        d = len(data)
        if duplicate:
            unique_data, duplicate_cnt = np.unique(data, return_counts=True, axis=0)
            duplicate_coeff = np.sum(duplicate_cnt * np.log(duplicate_cnt)) / len(data)
            free_energies = self.get_free_energy(unique_data)
            free_energies *= duplicate_cnt
            return float(log_z) + free_energies.sum() / d - np.log(d) + duplicate_coeff
        else:
            free_energies = self.get_free_energy(data)
            return float(log_z) + free_energies.sum() / d - np.log(d)
        
    def calc_kl_divergence_all_visible(self, data_v, data_h, duplicate=False):
        log_z = self.calc_log_partition_func()
        d = len(data_v)
        if duplicate:
            unique_data, duplicate_cnt = np.unique(np.hstack([data_v, data_h]), return_counts=True, axis=0)
            unique_data_v = unique_data[:, 0:self.nv]
            unique_data_h = unique_data[:, self.nv:]
            duplicate_coeff = np.sum(duplicate_cnt * np.log(duplicate_cnt)) / len(data_h)
            energies = self.get_energy(unique_data_v, unique_data_h)
            energies *= duplicate_cnt
            return float(log_z) + energies.sum() / d - np.log(d) + duplicate_coeff
        else:
            energies = self.get_energy(data_v, data_h)
            return float(log_z) + energies.sum() / d - np.log(d) 


    def calc_mpf_preprocess(self, data):
        flipped = (data.reshape(-1, 1, self.nv) * np.ones((1, self.nv, 1)) + np.eye(self.nv)) % 2
        free_energies = self.get_free_energy(data).reshape(-1, 1)
        free_energies_d = self.get_free_energy(flipped.reshape(-1, self.nv)).reshape(-1, self.nv)
        return flipped, free_energies, free_energies_d

    def calc_mpf_cost(self, data):
        _, free_energies, free_energies_d = self.calc_mpf_preprocess(data)
        return np.sum(np.exp((free_energies - free_energies_d)/2)) / len(data)

    def get_energy(self, v, h):
        return self.get_weight_energy(v, h) + self.get_visible_bias_energy(v) + self.get_hidden_bias_energy(h)

    def get_weight_energy(self, v, h):
        return -((v @ self.w).reshape(-1, 1, self.nh) @ h.reshape(-1, self.nh, 1)).ravel()

    def get_visible_bias_energy(self, v):
        return -self.bv @ v.T

    def get_hidden_bias_energy(self, h):
        return -self.bh @ h.T

    def get_free_energy(self, v):
        return -(v @ self.bv.reshape(-1, 1)).ravel() - np.sum(np.log(1 + np.exp(self.bh + (v @ self.w))), axis=1)

    def get_expect_hidden(self, v):
        return sigmoid(self.bh + (v @ self.w), np)

    def get_expect_visible(self, h):
        return sigmoid(self.bv + (h @ self.w.T), np)

    def get_hidden(self, v):
        ph = self.get_expect_hidden(v)
        return (ph > np.random.rand(self.nh)).astype('float64'), ph

    def get_visible(self, h):
        pv = self.get_expect_visible(h)
        return (pv > np.random.rand(self.nv)).astype('float64'), pv

    def train_cd(self, data, t=1):
        np.random.shuffle(data)
        size = (len(data) // self.batch_size) * self.batch_size
        for v0 in data[:size, :].reshape(-1, self.batch_size, self.nv):
            h0, ph0 = self.get_hidden(v0)
            hn = h0.copy()
            for _ in range(t):
                vn, pvn = self.get_visible(hn)
                hn, phn = self.get_hidden(vn)
            self.update_cd(v0, vn, ph0, phn)

    def train_cd_all_visible(self, data_v, data_h, t=1):
        size = (len(data_v) // self.batch_size) * self.batch_size
        p = np.random.permutation(len(data_v))
        data_v = data_v[p]
        data_h = data_h[p]
        for v0, h0 in zip(data_v[:size, :].reshape(-1, self.batch_size, self.nv), data_h[:size, :].reshape(-1, self.batch_size, self.nh)):
            hn = h0.copy()
            for _ in range(t):
                vn, pvn = self.get_visible(hn)
                hn, phn = self.get_hidden(vn)
            self.update_cd(v0, vn, h0, hn)

    def update_cd(self, v0, vn, h0, hn):
        self.w += self.optimizer_w((v0.reshape(self.batch_size, -1, 1) @ h0.reshape(self.batch_size, 1, -1) - vn.reshape(self.batch_size, -1, 1) @ hn.reshape(self.batch_size, 1, -1)).sum(axis=0))
        self.bv += self.optimizer_bv((v0 - vn).sum(axis=0))
        self.bh += self.optimizer_bh((h0 - hn).sum(axis=0))
        self.apply_limits()

    def get_ising(self):
        j = np.zeros((self.nv + self.nh, self.nv + self.nh))
        j[:self.nv, self.nv:] = -self.w / 4
        h = np.zeros(self.nv + self.nh)
        h[:self.nv] -= self.bv / 2 + self.w.sum(axis=1) / 4
        h[self.nv:] -= self.bh / 2 + self.w.sum(axis=0) / 4
        return j, h

    def get_mean_field(self, n=0):
        mu_v = np.random.rand(self.nv)
        mu_h = np.random.rand(self.nh)

        for i in range(n):
            mu_v = self.get_expect_visible(mu_h)
            mu_h = self.get_expect_hidden(mu_v)

        return mu_v, mu_h

    def apply_limits(self):
        if self.j_limit:
            if self.h_limit:
                w_min = -self.j_limit[1] * 4
                w_max = -self.j_limit[0] * 4
                self.w = self.w.clip(w_min, w_max)
                tmp_w = self.w.sum(axis=1) / 4
                bv_min = -2 * (self.h_limit[1] + tmp_w)
                bv_max = -2 * (self.h_limit[0] + tmp_w)
                self.bv = self.bv.clip(bv_min, bv_max)
                tmp_w = self.w.sum(axis=0) / 4
                bh_min = -2 * (self.h_limit[1] + tmp_w)
                bh_max = -2 * (self.h_limit[0] + tmp_w)
                self.bh = self.bh.clip(bh_min, bh_max)
                j, h = self.get_ising()
                pass

    def train_sa(self, data, mcmc_iter=100, burn_in=50):
        first_spin = [-1] * (self.nv + self.nh)
        np.random.shuffle(data)
        size = (len(data) // self.batch_size) * self.batch_size
        for vd in data[:size, :].reshape(-1, self.batch_size, self.nv):
            j, h = self.get_ising()
            
            graph = cxxjij.graph.Dense(self.nv + self.nh)
            for iv in range(self.nv):
                for ih in range(self.nv, self.nv + self.nh):
                    graph[iv, ih] = j[iv, ih]
            for i in range(self.nv + self.nh):
                graph[i] = h[i]

            self.spins = []
            system = cxxjij.system.make_classical_ising(first_spin, graph)
            cxxjij.algorithm.Algorithm_SingleSpinFlip_run(system, 0, [(1.0, mcmc_iter)], self.mcmc_callback)

            #Update
            self.spins = np.array(self.spins)
            vs = ((self.spins[burn_in:, :self.nv] + 1) / 2)
            hs = ((self.spins[burn_in:, self.nv:] + 1) / 2)
            self.update_gibbs(vd, vs, hs)

            first_spin = list(self.spins[-1].astype(np.int64))

    def train_mpf(self, data):  
        np.random.shuffle(data)
        size = (len(data) // self.batch_size) * self.batch_size
        for v0 in data[:size, :].reshape(-1, self.batch_size, self.nv):
            flipped, free_energy, free_energy_d = self.calc_mpf_preprocess(v0)
            diff_energy_exp = np.exp((free_energy - free_energy_d) / 2).ravel()
            dbv = np.sum(((-v0.reshape(-1, 1, self.nv) + flipped) / 2).reshape(-1, self.nv).transpose(1, 0) * diff_energy_exp, axis=1) / len(v0)
            tmp_sig = sigmoid(-v0 @ self.w - self.bh, np) - 1
            tmp_sig_d = sigmoid(-flipped @ self.w - self.bh, np) - 1
            dbh = np.sum(((tmp_sig.reshape(-1, 1, self.nh) - tmp_sig_d) / 2).reshape(-1, self.nh).transpose(1, 0) * diff_energy_exp, axis=1) / len(v0)
            tmp_dw = v0.reshape(-1, self.nv, 1) @ tmp_sig.reshape(-1, 1, self.nh)
            tmp_dw_d = flipped.reshape(-1, self.nv, self.nv, 1) @ tmp_sig_d.reshape(-1, self.nv, 1, self.nh)
            dw = np.sum(((tmp_dw.reshape(-1, 1, self.nv, self.nh) - tmp_dw_d) / 2).reshape(-1, self.nv, self.nh).transpose(1, 2, 0) * diff_energy_exp, axis=2) / len(v0)
            self.bv += self.optimizer_bv(-dbv)
            self.bh += self.optimizer_bh(-dbh)
            self.w += self.optimizer_w(-dw)
            self.apply_limits()
        pass

    def update_gibbs(self, vd, vs, hs):
        vd = vd.reshape(-1, self.nv)
        phd = self.get_expect_hidden(vd)
        self.bh += self.optimizer_bh((phd - np.average(hs, axis=0)).sum(axis=0))
        self.bv += self.optimizer_bv((vd - np.average(vs, axis=0)).sum(axis=0))
        self.w += self.optimizer_w((vd.reshape(-1, self.nv, 1) @ phd.reshape(-1, 1, self.nh) - np.average(vs.reshape(vs.shape[0], -1, 1) @ hs.reshape(vs.shape[0], 1, -1), axis=0)).sum(axis=0))
        self.apply_limits()

    def train_gibbs(self, data, mcmc_iter=50, burn_in=5):
        v = np.zeros((1, self.nv))
        np.random.shuffle(data)
        size = (len(data) // self.batch_size) * self.batch_size
        for vd in data[:size, :].reshape(-1, self.batch_size, self.nv):
            vs = []
            hs = []
            for i in range(mcmc_iter):
                h, _ = self.get_hidden(v)
                v, _ = self.get_visible(h)
                if i >= burn_in:
                    vs.append(v)
                    hs.append(h)
            vs = np.array(vs)
            hs = np.array(hs)
            self.update_gibbs(vd, vs, hs)

    def mcmc_callback(self, system, beta):
        self.spins.append(system.spin[0:system.num_spins].copy())

    def load(self, path):
        params = np.load(path)
        self.w = params['w']
        self.bv = params['bv']
        self.bh = params['bh']
        self.nv = len(self.bv)
        self.nh = len(self.bh)

    def save(self, path):
        np.savez(path, w=self.w, bv=self.bv, bh=self.bh)

class RBMStaticEstimator(RBM):
    def __init__(self, nv, nh, calib_optimizer='sgd', calib_opt_params={'lr': 0.000000001, 'decay': 1}, gpu=False):
        super().__init__(nv, nh, gpu=gpu)
        self.p = 1
        self.q = 1
        self.r = 1
        self.optimizer_p = select_optimizer(calib_optimizer, calib_opt_params)
        self.optimizer_q = select_optimizer(calib_optimizer, calib_opt_params)
        self.optimizer_r = select_optimizer(calib_optimizer, calib_opt_params)

    def get_cd_samples(self, vs, hs, cd_k):
        org_w = self.w   #copy is not needed
        org_bv = self.bv
        org_bh = self.bh
        
        self.w = self.w * self.p
        self.bv = self.bv * self.q
        self.bh = self.bh * self.r

        vn = vs.copy()
        hn = hs.copy()
        for __ in range(cd_k):
            vn, pvn = self.get_visible(hn)
            hn, phn = self.get_hidden(vn)

        self.w = org_w
        self.bv = org_bv
        self.bh = org_bh

        return vn, hn

    def estimate_one(self, vs, hs, cd_k=8):    #Update beta
        vn, hn = self.get_cd_samples(vs, hs, cd_k)
        grad = ((vs @ self.w).reshape(-1, 1, self.nh) @ hs.reshape(-1, self.nh, 1) - (vn @ self.w).reshape(-1, 1, self.nh) @ hn.reshape(-1, self.nh, 1)).sum()
        grad += (self.bv @ vs.T - self.bv @ vn.T).sum()
        grad += (self.bh @ hs.T - self.bh @ hn.T).sum()
        self.p += self.optimizer_p(grad)
        self.q = self.p
        self.r = self.p
        
    def estimate(self, vs, hs, cd_k=8):   #Update p, q, r
        vn, hn = self.get_cd_samples(vs, hs, cd_k)
        self.p += self.optimizer_p(((vs @ self.w).reshape(-1, 1, self.nh) @ hs.reshape(-1, self.nh, 1) - (vn @ self.w).reshape(-1, 1, self.nh) @ hn.reshape(-1, self.nh, 1)).sum())
        self.q += self.optimizer_q((self.bv @ vs.T - self.bv @ vn.T).sum())
        self.r += self.optimizer_r((self.bh @ hs.T - self.bh @ hn.T).sum())

    def estimate_all(self, vs, hs, cd_k=8):   #Update p, q, r
        vn, hn = self.get_cd_samples(vs, hs, cd_k)
        self.p += self.optimizer_p(((vs @ self.w).reshape(-1, 1, self.nh) @ hs.reshape(-1, self.nh, 1) - (vn @ self.w).reshape(-1, 1, self.nh) @ hn.reshape(-1, self.nh, 1)).sum())
        self.qs += self.optimizer_q((self.bv * vs - self.bv * vn).sum(axis=0))
        self.rs += self.optimizer_r((self.bh * hs - self.bh * hn).sum(axis=0))

    def calc_kl_divergence_all_visible(self, data_v, data_h, duplicate=False):
        org_w = self.w   #copy is not needed
        org_bv = self.bv
        org_bh = self.bh
        
        self.w = self.w * self.p
        self.bv = self.bv * self.q
        self.bh = self.bh * self.r
        kl = super().calc_kl_divergence_all_visible(data_v, data_h, duplicate)

        self.w = org_w
        self.bv = org_bv
        self.bh = org_bh
        
        return kl

class RBMStaticEstimatorAllBias(RBMStaticEstimator):
    def __init__(self, nv, nh, calib_optimizer='sgd', calib_opt_params={'lr': 0.000000001, 'decay': 1}, gpu=False):
        super().__init__(nv, nh, calib_optimizer, calib_opt_params, gpu)
        self.q = np.ones(nv)
        self.r = np.ones(nh)

    def load(self, path):
        super().load(path)
        self.q = np.ones(self.nv)
        self.r = np.ones(self.nh)

    def estimate(self, vs, hs, cd_k=8):   #Update p, q, r
        vn, hn = self.get_cd_samples(vs, hs, cd_k)
        self.p += self.optimizer_p(((vs @ self.w).reshape(-1, 1, self.nh) @ hs.reshape(-1, self.nh, 1) - (vn @ self.w).reshape(-1, 1, self.nh) @ hn.reshape(-1, self.nh, 1)).sum())
        self.q += self.optimizer_q((self.bv * vs - self.bv * vn).sum(axis=0))
        self.r += self.optimizer_r((self.bh * hs - self.bh * hn).sum(axis=0))

class RBMStaticEstimatorAllBias(RBMStaticEstimator):
    def __init__(self, nv, nh, calib_optimizer='sgd', calib_opt_params={'lr': 0.000000001, 'decay': 1}, gpu=False):
        super().__init__(nv, nh, calib_optimizer, calib_opt_params, gpu)
        self.q = np.ones(nv)
        self.r = np.ones(nh)

    def load(self, path):
        super().load(path)
        self.q = np.ones(self.nv)
        self.r = np.ones(self.nh)

    def estimate(self, vs, hs, cd_k=8):   #Update p, q, r
        vn, hn = self.get_cd_samples(vs, hs, cd_k)
        self.p += self.optimizer_p(((vs @ self.w).reshape(-1, 1, self.nh) @ hs.reshape(-1, self.nh, 1) - (vn @ self.w).reshape(-1, 1, self.nh) @ hn.reshape(-1, self.nh, 1)).sum())
        self.q += self.optimizer_q((self.bv * vs - self.bv * vn).sum(axis=0))
        self.r += self.optimizer_r((self.bh * hs - self.bh * hn).sum(axis=0))

class RBMStaticEstimatorFull(RBMStaticEstimator):
    def __init__(self, nv, nh, calib_optimizer='sgd', calib_opt_params={'lr': 0.000000001, 'decay': 1}, gpu=False):
        super().__init__(nv, nh, calib_optimizer, calib_opt_params, gpu)
        self.p = np.ones((nv, nh))
        self.q = np.ones(nv)
        self.r = np.ones(nh)

    def load(self, path):
        super().load(path)
        self.p = np.ones((self.nv, self.nh))
        self.q = np.ones(self.nv)
        self.r = np.ones(self.nh)

    def estimate(self, vs, hs, cd_k=8):   #Update p, q, r
        vn, hn = self.get_cd_samples(vs, hs, cd_k)
        self.p += self.optimizer_p((self.w * (vs.reshape(-1, self.nv, 1) @ hs.reshape(-1, 1, self.nh) - vn.reshape(-1, self.nv, 1) @ hn.reshape(-1, 1, self.nh))).sum(axis=0))
        self.q += self.optimizer_q((self.bv * vs - self.bv * vn).sum(axis=0))
        self.r += self.optimizer_r((self.bh * hs - self.bh * hn).sum(axis=0))

class RBMCalibWithSampler(RBM): #This is abstract
    def __init__(self, nv, nh, optimizer='momentum', opt_params={ 'lr': 0.01,'alpha': 0.9,'decay': 1}, batch_size=100, gpu=False, j_limit=None, h_limit=None, calib_optimizer='sgd', calib_w_params={'lr': 0.000001, 'decay': 1}, calib_bv_params={'lr': 0.000001, 'decay': 1}, calib_bh_params={'lr': 0.000001, 'decay': 1}):
        super().__init__(nv, nh, optimizer, opt_params, batch_size, gpu, j_limit, h_limit)
        self.init_calib_params()
        self.optimizer_alpha = select_optimizer(calib_optimizer, calib_w_params)
        self.optimizer_beta_v = select_optimizer(calib_optimizer, calib_bv_params)
        self.optimizer_beta_h = select_optimizer(calib_optimizer, calib_bh_params)
        self.samples = []

    def load(self, path):
        super().load(path)
        self.init_calib_params()

    def set_sampler(self, sampler):
        self.sampler = sampler

    def train_sample(self, data, ns=1000, calib_k=1, calib_n=1, mode=''):
        np.random.shuffle(data)
        size = (len(data) // self.batch_size) * self.batch_size
        for vd in data[:size, :].reshape(-1, self.batch_size, self.nv):
            self.sampler.set_weights(self.w / self.alpha, self.bv / self.beta_v, self.bh / self.beta_h)
            vs, hs = self.sampler.get_samples(ns)
            self.calibrate(vs, hs, cd_k=calib_k, n=calib_n, mode=mode)
            self.update_gibbs(vd, vs, hs)
            self.apply_limits()

    def calibrate(self, vs, hs, cd_k=1, n=1, mode=''):
        self.tmp_alpha = np.ones_like(self.alpha)
        self.tmp_beta_v = np.ones_like(self.beta_v)
        self.tmp_beta_h = np.ones_like(self.beta_h)

        org_w = self.w
        org_bv = self.bv
        org_bh = self.bh

        for _ in range(n):
            self.w = self.tmp_alpha * org_w
            self.bv = self.tmp_beta_v * org_bv
            self.bh = self.tmp_beta_h * org_bh

            hn = hs.copy()
            vn = vs.copy()
            for __ in range(cd_k):
                tmp_vn, pvn = self.get_visible(hn)
                hn, phn = self.get_hidden(vn)
                vn = tmp_vn

            self.w = org_w
            self.bv = org_bv
            self.bh = org_bh
            self.update_calib_params(vs, hs, vn, hn, mode)


        self.alpha *= self.tmp_alpha
        self.beta_v *= self.tmp_beta_v
        self.beta_h *= self.tmp_beta_h

class RBMFullCalibWithSampler(RBMCalibWithSampler):    
    def init_calib_params(self):
        self.alpha = np.ones_like(self.w)
        self.beta_v = np.ones_like(self.bv)
        self.beta_h = np.ones_like(self.bh)

    def update_calib_params(self, vs, hs, vn, hn):
        self.tmp_alpha += self.optimizer_alpha(self.w * ((vs.reshape(-1, self.nv, 1) @ hs.reshape(-1, 1, self.nh) - vn.reshape(-1, self.nv, 1) @ hn.reshape(-1, 1, self.nh)).sum(axis=0)))
        self.tmp_beta_v += self.optimizer_beta_v(self.bv * ((vs - vn).sum(axis=0)))
        self.tmp_beta_h += self.optimizer_beta_h(self.bh * ((hs - hn).sum(axis=0)))

class RBMEachCoefCalibWithSampler(RBMCalibWithSampler):
    def init_calib_params(self):
        self.alpha = 1.
        self.beta_v = 1.
        self.beta_h = 1.

    def update_calib_params(self, vs, hs, vn, hn, mode='each'):
        if mode == 'one':
            self.tmp_alpha += self.optimizer_alpha((self.get_energy(vn, hn) - self.get_energy(vs, hs)).sum())
            self.tmp_beta_v = self.tmp_alpha
            self.tmp_beta_h = self.tmp_alpha
        else:
            self.tmp_alpha += self.optimizer_alpha((self.get_weight_energy(vn, hn) - self.get_weight_energy(vs, hs)).sum())
            self.tmp_beta_v += self.optimizer_beta_v((self.get_visible_bias_energy(vn) - self.get_visible_bias_energy(vs)).sum())
            self.tmp_beta_h += self.optimizer_beta_h((self.get_hidden_bias_energy(hn) - self.get_hidden_bias_energy(hs)).sum())

class RBMAllBiasCalibWithSampler(RBMCalibWithSampler):
    def init_calib_params(self):
        self.alpha = 1.
        self.beta_v = np.ones_like(self.bv)
        self.beta_h = np.ones_like(self.bh)

    def update_calib_params(self, vs, hs, vn, hn, mode='all'):
        if mode == 'one':
            d = self.optimizer_alpha((self.get_energy(vn, hn) - self.get_energy(vs, hs)).sum())
            self.tmp_alpha += d
            self.tmp_beta_v[:] = self.tmp_alpha
            self.tmp_beta_h[:] = self.tmp_alpha
        elif mode == 'each':
            self.tmp_alpha += self.optimizer_alpha((self.get_weight_energy(vn, hn) - self.get_weight_energy(vs, hs)).sum())
            self.tmp_beta_v += self.optimizer_beta_v((self.get_visible_bias_energy(vn) - self.get_visible_bias_energy(vs)).sum())
            self.tmp_beta_h += self.optimizer_beta_h((self.get_hidden_bias_energy(hn) - self.get_hidden_bias_energy(hs)).sum())
        else:
            self.tmp_alpha += self.optimizer_alpha((self.get_weight_energy(vn, hn) - self.get_weight_energy(vs, hs)).sum())
            self.tmp_beta_v += self.optimizer_beta_v(self.bv * ((vs - vn).sum(axis=0)))
            self.tmp_beta_h += self.optimizer_beta_h(self.bh * ((hs - hn).sum(axis=0)))

class RBMOneCoefCalibWithSampler(RBMCalibWithSampler):
    def init_calib_params(self):
        self.alpha = 1.
        self.beta_v = 1.
        self.beta_h = 1.

    def update_calib_params(self, vs, hs, vn, hn, mode=''):
        self.tmp_alpha += self.optimizer_alpha((self.get_energy(vn, hn) - self.get_energy(vs, hs)).sum())
        self.tmp_beta_v = self.tmp_alpha
        self.tmp_beta_h = self.tmp_alpha

class RBMSampler:
    def __init__(self, nv, nh):
        self.rbm = RBM(nv, nh)

    def set_weights(self, w, bv, bh):
        self.rbm.w = w
        self.rbm.bv = bv
        self.rbm.bh = bh

    def get_samples(self, n, burn_in=5):
        vs = []
        hs = []
        v = np.zeros((1, self.rbm.nv))
        for i in range(n + burn_in):
            h, ph = self.rbm.get_hidden(v)
            v, pv = self.rbm.get_visible(h)
            if i >= burn_in:
                vs.append(v[0])
                hs.append(h[0])
        return np.array(vs), np.array(hs)

class RBMSamplerWithFullError(RBMSampler):
    def __init__(self, nv, nh, error_strength=0.01, error_offset=0):
        super().__init__(nv, nh)
        self.w_error = np.random.normal(1, error_strength, self.rbm.w.shape) + error_offset
        self.bv_error = np.random.normal(1, error_strength, self.rbm.bv.shape) + error_offset
        self.bh_error = np.random.normal(1, error_strength, self.rbm.bh.shape) + error_offset

    def set_weights(self, w, bv, bh):
        self.rbm.w = w * self.w_error
        self.rbm.bv = bv * self.bv_error
        self.rbm.bh = bh * self.bh_error
        
class RBMSamplerWithEachCoefError(RBMSamplerWithFullError):
    def __init__(self, nv, nh, error_strength=0.01, error_offset=0):
        super().__init__(nv, nh)
        self.w_error = np.random.normal(1, error_strength) + error_offset
        self.bv_error = np.random.normal(1, error_strength) + error_offset
        self.bh_error = np.random.normal(1, error_strength) + error_offset

class RBMSamplerWithAllBiasError(RBMSamplerWithFullError):
    def __init__(self, nv, nh, error_strength=0.01, error_offset_w=0, error_offset_bv=0, error_offset_bh=0):
        super().__init__(nv, nh)
        self.w_error = np.random.normal(1, error_strength) + error_offset_w
        self.bv_error = np.random.normal(1, error_strength, self.rbm.bv.shape) + error_offset_bv
        self.bh_error = np.random.normal(1, error_strength, self.rbm.bh.shape) + error_offset_bh

class DW32x8Sampler:
    DISP_INSPECTOR = True
    def __init__(self, embedding):
        self.dummy_rbm = RBM(32, 8)
        self.j = {}
        self.h = {}
        self.sampler = DWaveSampler(solver=self.SOLVER, token=os.environ.get("dwave_token"))
        self.sampler = FixedEmbeddingComposite(self.sampler, embedding=embedding)

    def set_weights(self, w, bv, bh):
        self.dummy_rbm.w = w
        self.dummy_rbm.bv = bv
        self.dummy_rbm.bh = bh
        tmp_j, tmp_h = self.dummy_rbm.get_ising()
        tmp_j = tmp_j.clip(*self.J_LIMIT)
        tmp_h = tmp_h.clip(*self.H_LIMIT)

        for grp in range(self.grp_size):
            for row in range(0, self.dummy_rbm.nv):
                for col in range(self.dummy_rbm.nv, self.dummy_rbm.nv + self.dummy_rbm.nh):
                    self.j[((grp, row), (grp, col))] = tmp_j[row, col]

            for i in range(self.dummy_rbm.nv + self.dummy_rbm.nh):
                self.h[(grp, i)] = tmp_h[i]

    def get_samples(self, n):
        num_reads = int(np.ceil(n / self.grp_size))
        samples = []
        while num_reads > 0:
            tmp_num_reads = min(num_reads, 10000)
            self.response = self.sampler.sample_ising(self.h, self.j, num_reads=tmp_num_reads, auto_scale=False)
            num_reads -= tmp_num_reads          
            for r in self.response.record:
                samples.append(r[0])

        samples = np.array(samples).reshape(-1, self.dummy_rbm.nv + self.dummy_rbm.nh)
        samples = (samples[:n, :] + 1) / 2
        return samples[:, :self.dummy_rbm.nv], samples[:, self.dummy_rbm.nv:]

class DW2000_32x8Sampler(DW32x8Sampler):
    J_LIMIT = [-1, 1]
    H_LIMIT = [-2, 2]
    START_NODES = [64, 128, 320, 576, 640, 832, 896, 1088, 1152, 1344, 1408, 1856] #for D-WAVE 2000Q_6
    SOLVER = 'DW_2000Q_6'

    def __init__(self):
        self.grp_size = len(self.START_NODES)
        embedding = {}
        for grp, start in enumerate(self.START_NODES):
            for cell in range(8):
                for vis in range(4):
                    idx = cell * 4 + vis
                    embedding[(grp, idx)] = [start + cell * 8 + vis, start + 128 + cell * 8 + vis]
            for cell in range(2):
                for hid in range(4):
                    idx = cell * 4 + hid + 32
                    embedding[(grp, idx)] = [start + 4 + cell * 128 + hid + i * 8 for i in range(8)]
        super().__init__(embedding)

class DWAdvantage_32x8Sampler(DW32x8Sampler):
    J_LIMIT = [-1, 1]
    H_LIMIT = [-4, 4]
    START_CELLS_HOR = [(0, 0, 1), (3, 0, 0), (3, 0, 1), (3, 0, 2), (5, 0, 1), (5, 0, 2), (8, 7, 0), (8, 7, 1), (8, 7, 2), (11, 7, 0), (11, 7, 1), (11, 7, 2), (13, 7, 0), (13, 7, 1), (13, 7, 2)] #for Advantage_system6.1
    START_CELLS_VER = [(0, 8, 2), (0, 11, 1), (0, 13, 0), (0, 13, 1), (0, 13, 2), (7, 1, 1), (7, 1, 2), (7, 3, 0), (7, 3, 1)]#, (7, 5, 0), (7, 5, 1), (7, 5, 2)]
    SOLVER = 'Advantage_system6.1'

    @staticmethod
    def get_ver_node(r, c, grp, idx):
        return r * 180 + grp * 60 + c + 3000 + 15 * idx

    @staticmethod
    def get_hor_node(r, c, grp, idx):
        return c * 180 + (2 - grp) * 60 + r + 60 + 15 * idx

    def __init__(self):
        self.grp_size = len(self.START_CELLS_HOR + self.START_CELLS_VER)
        embedding = {}
        for grp, start in enumerate(self.START_CELLS_HOR + self.START_CELLS_VER):
            for cell in range(8):
                for vis in range(4):
                    idx = cell * 4 + vis
                    if grp < len(self.START_CELLS_HOR):
                        embedding[(grp, idx)] = [self.get_hor_node(start[0], start[1] + cell, start[2], vis), self.get_hor_node(start[0] + 1, start[1] + cell, start[2], vis)]
                    else:
                        embedding[(grp, idx)] = [self.get_ver_node(start[0] + cell, start[1], start[2], vis), self.get_ver_node(start[0] + cell, start[1] + 1, start[2], vis)]
            for cell in range(2):
                for hid in range(4):
                    idx = cell * 4 + hid + 32
                    if grp < len(self.START_CELLS_HOR):
                        embedding[(grp, idx)] = [self.get_ver_node(start[0] + cell, start[1] + i, start[2], hid) for i in range(8)]
                    else:
                        embedding[(grp, idx)] = [self.get_hor_node(start[0] + i, start[1] + cell, start[2], hid) for i in range(8)]
        super().__init__(embedding)       
