'''
Auxiliary functions for EM algorithm testing.

Author: Artem Sevastopolsky, 2015
'''

import numpy as np
import scipy.linalg
import scipy.misc
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing
import sklearn.metrics
import sklearn.cross_validation
from numba import jit, autojit
import numexpr


def gen_dataset(N, D, K):
    mean_range = 100
    
    mean = np.random.rand(K, D) * mean_range - mean_range / 2
    cov = np.eye(D)
    cls_size = N // K
    X = np.vstack((np.random.multivariate_normal(mean[k], cov, 
                                                 size=(cls_size if k != K - 1 else cls_size + N % K))
                   for k in range(K)))
    y = np.empty(N)
    for k in range(K - 1):
        y[cls_size * k : cls_size * (k + 1)] = k
    y[cls_size * (K - 1):] = K - 1
    
    # randomly shuffling data set
    idx = np.arange(N)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    return (X, y)


def show_err_pics(y_pred, y_true, pic_shape, pic_to_show=3):
    all_errors = errors(y_true, y_pred)
    all_errors_sum = [all_errors[i].sum(axis=1) for i in range(4)]
    
    for err_type in range(2):
        fig, ax_arr = plt.subplots(1, pic_to_show, figsize=(25, 25))
        # showing objects with maximal type I or type II error
        samples_idx = np.argsort(all_errors_sum[err_type + 1])[-pic_to_show:]
        for i, idx in enumerate(samples_idx):
            errors_map = np.empty((pic_shape[0] * pic_shape[1], 3))
            for k in range(3):
                errors_map[:, k] = y_pred[idx]
            errors_map[(y_pred[idx] == 0) & (y_true[idx] == 1)] = np.array([0, 0, 1])    # FN - type II errors - blue
            errors_map[(y_pred[idx] == 1) & (y_true[idx] == 0)] = np.array([0, 1, 0])    # FP - type I errors - green
            ax_arr[i].imshow(errors_map.reshape((pic_shape[0], pic_shape[1], 3)))
            ax_arr[i].axis('off')


def errors(y_true, y_pred):
    N, D = y_true.shape
    
    fp = np.zeros((N, D), dtype=bool)
    fn = np.zeros((N, D), dtype=bool)
    tp = np.zeros((N, D), dtype=bool)
    tn = np.zeros((N, D), dtype=bool)
    for i in range(N):
        tp[i, :] = (y_pred[i] == 1) & (y_true[i] == 1)
        fp[i, :] = (y_pred[i] == 1) & (y_true[i] == 0)
        fn[i, :] = (y_pred[i] == 0) & (y_true[i] == 1)
        tn[i, :] = (y_pred[i] == 0) & (y_true[i] == 0)
    # FP - type I errors, FN - type II errors
    return (tp, fp, fn, tn)
