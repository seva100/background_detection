# -*- coding: utf-8 -*-
"""
Created on Sun Sep 06 22:10:57 2015

@author: Artem Sevastopolsky
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from matplotlib.pyplot import imshow, figure


def load_image(path):
    # Loads image `path` and returns array of its pixels in range [0, 1]
    return np.asarray(Image.open(path)) / 255.0


def save_image(path, img):
    tmp = np.asarray(img * 255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)


def show_image(img, fig_size=(8, 8)):
    figure(figsize=fig_size)
    imshow(img, cmap=cm.Greys_r)


def load_set(folder, shuffle=False, n_first=None):
    img_list = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    if shuffle:
        np.random.shuffle(img_list)
    if n_first is None:
        n_first = len(img_list)
    else:
        img_list = img_list[:n_first]
    
    data = []
    for img_fn in img_list:
        img = load_image(img_fn)
        data.append(img)
    return data, img_list


def load_set_eq_shape(folder, shuffle=False, n_first=None, skip=0):
    img_list = sorted(glob.glob(os.path.join(folder, '*.jpg')) + 
                      glob.glob(os.path.join(folder, '*.png')))
    if shuffle:
        np.random.shuffle(img_list)
    if n_first is None:
        n_first = len(img_list)
    img_list = img_list[skip:skip + n_first]
    if n_first == 0 or not img_list:
        return np.empty(0), []
    
    first_img = load_image(img_list[0])
    img_size = first_img.shape[0] * first_img.shape[1]
    img_ch = first_img.shape[2]
    data = np.empty((len(img_list), img_size, img_ch))
    data[0, :, :] = first_img.reshape(img_size, img_ch)
    for i in range(1, len(img_list)):
        img = load_image(img_list[i])
        assert img.shape == first_img.shape
        data[i] = img.reshape((img_size, img_ch))
    return data, img_list


def load_set_eq_shape_gs(folder, shuffle=False, n_first=None, skip=0):
    img_list = sorted(glob.glob(os.path.join(folder, '*.jpg')) + \
                      glob.glob(os.path.join(folder, '*.png')))
    if shuffle:
        np.random.shuffle(img_list)
    if n_first is None:
        n_first = len(img_list)
    img_list = img_list[skip:skip + n_first]
    if n_first == 0 or not img_list:
        return np.empty(0), []
    
    first_img = load_image(img_list[0])
    img_size = first_img.shape[0] * first_img.shape[1]
    data = np.empty((len(img_list), img_size))
    data[0, :] = first_img.reshape(img_size)
    for i in range(1, len(img_list)):
        img = load_image(img_list[i])
        assert img.shape == first_img.shape
        data[i] = img.reshape(img_size)
    return data, img_list


def rgb_to_grayscale_3d(rgb_img):
    gs_img = 0.2126 * rgb_img[:, :, 0] + 0.7152 * rgb_img[:, :, 1] + \
        0.0722 * rgb_img[:, :, 2]
    return gs_img


def rgb_to_grayscale_2d(rgb_line):
    gs_img = 0.2126 * rgb_line[:, 0] + 0.7152 * rgb_line[:, 1] + \
        0.0722 * rgb_line[:, 2]
    return gs_img
