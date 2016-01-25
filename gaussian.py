'''
Density estimation methods.

Author: Artem Sevastopolsky, 2015
'''

import numpy as np
import scipy as sp
import scipy.linalg
import scipy.misc
import scipy.stats
import sklearn
import sklearn.preprocessing
import sklearn.metrics
from numba import jit
import numexpr

### Estimation with 1D Gaussian.

def gauss_1d_params(X):
    sigma_low_limit = 0.1
    
    N, D = X.shape
    
    # maximum likelihood estimations of unknown parameters
    mu = X.mean(axis=0)
    sigma = np.maximum(np.sqrt(((X - mu) * (X - mu)).mean(axis=0)), 
                       np.full((1, D), sigma_low_limit)).ravel()
    return {'mu': mu, 'sigma': sigma}


def gauss_1d_predict(X, mu, sigma, threshold):
    # threshold can be either int or float or array-like for each pixel
    return np.abs(X - mu) > threshold * sigma


def gauss_1d_predict_proba(X, mu, sigma):
    return np.abs(X - mu) / sigma


### Estimation with adaptive 1D Gaussian.

def gauss_1d_adaptive_predict(X, mu, sigma, threshold, rho):
    sigma_low_limit = 0.1
    
    N, D = X.shape
    
    mu_cur, sigma_cur = mu.copy(), sigma.copy()
    pred = np.empty_like(X, dtype=int)
    for i in range(N):
        pred[i] = np.abs(X[i] - mu_cur) > threshold * sigma_cur
        mu_cur[pred[i] == 0] = rho * X[i, pred[i] == 0] + (1 - rho) * mu_cur[pred[i] == 0]
        sigma_cur[pred[i] == 0] = np.sqrt((X[i, pred[i] == 0] - mu_cur[pred[i] == 0]) ** 2 * rho + \
                                          (1 - rho) * sigma_cur[pred[i] == 0] ** 2)
        sigma_cur[pred[i] == 0] = np.maximum(sigma_cur[pred[i] == 0], np.full((pred[i] == 0).sum(), sigma_low_limit))
        #print('i: ', i)
        #print('mu_cur: ', mu_cur)
        #print('sigma_cur: ', sigma_cur)
    return pred


### Estimation with multivariate ND Gaussian.

def logpdf(X, mu, sigma, sigma_inv, D):
    ans = -0.5 * (np.dot((X - mu), sigma_inv) * (X - mu)).sum(axis=1)
    try:
        #scipy.linalg.cholesky(sigma, overwrite_a=True)
        chol = scipy.linalg.cholesky(sigma)
        ans -= 0.5 * D * np.log(2 * np.pi) + (np.log(np.abs(chol[np.arange(D), np.arange(D)]))).sum()
    except (sp.linalg.LinAlgError, ValueError):
        return float('inf')
    return ans


#@jit
def gauss_multivariate_params(X, diag=False):
    regn_eps = 1e-4
    
    N, D, C = X.shape
    
    # maximum likelihood estimations of unknown parameters
    mu = X.mean(axis=0)
    Xs = X - mu
    sigma = np.zeros((D, C, C), dtype=float)
    for d in range(D):
        sigma_add = Xs[:, d, :].T.dot(Xs[:, d, :])
        if not diag:
            sigma[d] += sigma_add
        else:
            sigma[d, np.arange(C), np.arange(C)] += sigma_add[np.arange(C), np.arange(C)]
        sigma[d, np.arange(C), np.arange(C)] += regn_eps
        #if diag:
        #    sigma[d][~np.eye(D).astype(bool)] = 0.0
    sigma /= N
    return {'mu': mu, 'sigma': sigma}


def gauss_multivariate_predict(X, mu, sigma, threshold, diag=False):
    N, D, C = X.shape
    
    if np.isclose(threshold, 0):
        return np.zeros((N, D))
    log_t = np.log(threshold)
    pred = np.empty((N, D))
    for d in range(D):
        if not diag:
            sigma_inv = np.linalg.pinv(sigma[d])
        else:
            sigma_inv = sigma[d].copy()
            sigma_inv[np.arange(C), np.arange(C)] = 1.0 / sigma_inv[np.arange(C), np.arange(C)]
        pred[:, d] = (logpdf(X[:, d, :], mu[d], sigma[d], sigma_inv, C) < log_t)
    return pred


def gauss_multivariate_predict_proba(X, mu, sigma, diag=False):
    N, D, C = X.shape
    
    pred_proba = np.empty((N, D), dtype=float)
    for d in range(D):
        if not diag:
            sigma_inv = np.linalg.pinv(sigma[d])
        else:
            sigma_inv = sigma[d].copy()
            sigma_inv[np.arange(C), np.arange(C)] = 1.0 / sigma_inv[np.arange(C), np.arange(C)]
        pred_proba[:, d] = 1.0 - logpdf(X[:, d, :], mu[d], sigma[d], sigma_inv, C)
    return pred_proba

