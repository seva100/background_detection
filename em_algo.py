'''
EM algorithm. Training and prediction stage.

Author: Artem Sevastopolsky, 2015
'''

import time
import numpy as np
import scipy as sp
import scipy.linalg
from numba import jit
import numexpr
# gaussian.py - self-written library
from gaussian import gauss_multivariate_predict_proba, logpdf


@jit
def log_likelihood(X, gamma, w, mu, sigma, sigma_inv):
    N, D = X.shape
    K = gamma.shape[1]
    
    ans = gamma.dot(np.log(w).reshape((K, 1))).sum()
    for k in range(K):
        #try:
            #ans += (gamma[:, k] * scipy.stats.multivariate_normal.logpdf(X, mu[k], sigma[k])).sum()
            ans += (gamma[:, k] * logpdf(X, mu[k], sigma[k], sigma_inv[k], D)).sum()
        #except ValueError:
        #    return float('inf')
    ans /= N    # dividing by constant does not change clustering but makes numbers smaller.
    return ans


def em_gauss_mix_params(X, K, diag=False, verbose=False):
    '''Implementation of EM algorithm for recovering 
    gaussian mixture model parameters.
    Arguments:
    X: array N x D - data set
    K: int - number of mixture components
    verbose: bool - whether or not to show cost (log likelihood) at each step
    '''
    sigma_regn = 1e-5
    
    N, D = X.shape
    gamma = np.empty((N, K), dtype=float)
    random_objs_idx = np.arange(N)
    np.random.shuffle(random_objs_idx)
    mu = X[random_objs_idx[:K]]
    sigma = np.zeros((K, D, D), dtype=float)
    sigma_inv = np.empty((K, D, D), dtype=float)
    for k in range(K):
        sigma[k] = np.eye(D)
        sigma_inv[k] = np.eye(D)
    #w = np.random.uniform(low=1.0, high=5.0, size=(K))
    w = np.full(K, 1.0 / K)
    #norm_observn = np.empty((N, K), dtype=float)
    exp_vals = np.empty((N, K))
    sigma_det = np.empty(K)
    
    cost = []
    prefer = None
    iter_time = []
    while len(cost) < 2 or (not np.isclose(cost[-1] - cost[-2], 0, atol=1e-4) and np.isfinite(cost[-1])):
    #while len(cost) < 2 or np.any(prefer != prefer_old):
        time_start = time.time()
        try:
            for k in range(K):
                # at first, exp_vals stores arguments of exp, not vals
                exp_vals[:, k] = -0.5 * (np.dot(X - mu[k], sigma_inv[k]) * (X - mu[k])).sum(axis=1)
            exp_vals -= exp_vals.max(axis=1).reshape((N, 1))
            exp_vals = np.exp(exp_vals)
            
            for k in range(K):
                sigma_det[k] = np.linalg.det(sigma[k])
            v = (w / sigma_det).reshape((K, 1))
            gamma = v.T * exp_vals / np.dot(exp_vals, v)
            
            w = np.mean(gamma, axis=0)
            
            for k in range(K):
                mu[k] = (gamma[:, k].reshape((N, 1)) * X).sum(axis=0) / gamma[:, k].sum()
                #if (np.isclose(gamma[:, k].sum(), 0) or not np.isfinite(gamma[:, k].sum())):
                #    print('incorrect sum in mu: ', k, gamma[:, k].sum())
            
            sigma[:, :, :] = 0.0
            for k in range(K):
                #for n in range(N):
                #    sigma[k] += gamma[n, k] * (X[n] - mu[k]).reshape((D, 1)).dot((X[n] - mu[k]).reshape((1, D)))
                sigma[k] = (gamma[:, k].reshape((N, 1)) * (X - mu[k])).T.dot((X - mu[k]))
                sigma[k] /= gamma[:, k].sum()
                # sigma matrix regularization
                sigma[k, np.arange(D), np.arange(D)] += sigma_regn
                if diag:
                    sigma[k][~np.eye(D).astype(bool)] = 0.0
            
            for k in range(K):
                sigma_inv[k] = np.linalg.inv(sigma[k])
            
            cost.append(log_likelihood(X, gamma, w, mu, sigma, sigma_inv))
        except (sp.linalg.LinAlgError, ValueError):
            # we can do nothing in such situation
            return {'w': w, 'mu': mu, 'sigma': sigma, 'gamma': gamma, 
                    'cost': cost, 'iter_time': np.array(iter_time).mean()}
        if verbose:
            print('cost: ', cost[-1])
        prefer = gamma.argmax(axis=1)
        
        time_end = time.time()
        iter_time.append(time_end - time_start)
    
    return {'w': w, 'mu': mu, 'sigma': sigma, 'gamma': gamma, 
            'cost': cost, 'iter_time': np.array(iter_time).mean()}
           

def em_gauss_mix_params_tries(X, K, tries=5, diag=False, verbose=False):
    N = X.shape[0]
    
    max_cost = max_cost_result = None
    for i in range(1, tries + 1):
        if verbose:
            print('Try #{}'.format(i))
        result = em_gauss_mix_params(X, K, diag, verbose)
        if max_cost is None or result['cost'] > max_cost:
            max_cost = result['cost']
            max_cost_result = result
    return max_cost_result


def em_predict(X, em_mu, em_sigma, em_w, N, D, K, threshold):
    pred_compn = [gauss_multivariate_predict_proba(X, em_mu[:, k, :], em_sigma[:, k, :])
                  for k in range(K)]
    pred_proba = np.zeros((N, D), dtype=float)
    for k in range(K):
        pred_proba += pred_compn[k] * em_w[:, k]
    pred = (pred_proba < threshold)
    
    return pred


def em_predict_proba(X, em_mu, em_sigma, em_w, N, D, K):
    pred_compn = [gauss_multivariate_predict_proba(X, em_mu[:, k, :], em_sigma[:, k, :])
                  for k in range(K)]
    pred_proba = np.zeros((N, D), dtype=float)
    for k in range(K):
        pred_proba += pred_compn[k] * em_w[:, k]
    
    return pred_proba
