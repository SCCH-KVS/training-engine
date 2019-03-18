#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 2/20/2019 1:12 PM $ 
# by : shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------

import os
import time
import cv2
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from numba import jit, vectorize, cuda


def one_hot_encode(labels):
    """
    Convert labels to one-hot
    :param labels:
    :return:
    """
    if not isinstance(labels[0], list):
        mx = max(labels)
        y = []
        for i in range(len(labels)):
            f = [0] * (mx + 1)
            f[labels[i]] = 1
            y.append(f)
        return y
    else:
        return labels


def one_to_onehot(label, max_label):
    y = [0 for i in range(max_label)]
    y[label] = 1
    return y


# @jit(parallel=True)
def resize_images_gpu(x, img_size):

    return cv2.resize(x, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    # if x.size == img_size[0:2]:
    #     return x
    # else:
    #     return np.rollaxis(np.array(x.resize(img_size[0:2],  Image.ANTIALIAS)), 1, 0)

    # return x

# @jit
def expand_dims(x, img_size):
    # return np.expand_dims(np.expand_dims(resize_images_gpu(x, img_size), axis=0), axis=-1)

    return np.expand_dims(np.expand_dims(x, axis=0), axis=-1)

# @jit
def resize_images_cpu(x, img_size):
    return cv2.resize(x, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)

# @jit(parallel=True)
def preprocess_image(img_size, im_file):

    img = cv2.imread(im_file, 0)
    image = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    # img = Image.open(im_file)
    # img = np.rollaxis(Image.open(im_file), 1, 0)
    # image = resize_images_gpu(img_size,img)
    return image


# @jit(parallel=True)
def bulk_process(img_size, im):
    # img_out = []
    # for im in im_list:
    #     img_out.append(np.expand_dims(preprocess_image(im, img_size), axis=-1))
    # return img_out
    return preprocess_image(img_size, im)


def path_walk(path):
    path_list = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.lower().endswith(tuple(['.jpg', '.jpeg', '.png'])):
                path_list.append(os.path.abspath(os.path.join(dirpath, f)))
    return path_list


def log_loss_accuracy(accuracy, accuracy_type, task_type, num_classes, multi_task):
    if isinstance(accuracy_type, list) and (len(accuracy_type) >= 2):
        acc_str = ''
        for k in range(len(accuracy_type)):
            acc_str += '{:s} '.format(accuracy_type[k])
            for i in range(num_classes):
                acc_str += 'lbl_{:d}: {:.3f} '.format(i, accuracy[k][i])
    else:
        if isinstance(accuracy_type, list):
            accuracy_type = accuracy_type[0]
        acc_str = '{:s}'.format(accuracy_type)
        if task_type == 'classification':
            if multi_task:
                acc_str += 'task_1: {:.3f} task_2: {:.3f} '.format(accuracy[0], accuracy[1])
            else:
                acc_str += 'lbl: {:.3f} '.format(accuracy)
        elif task_type == 'segmentation':
            acc_str += 'lbl: {:.3f} '.format(accuracy)
            # for i in range(num_classes):
            #     acc_str += 'lbl_{:d}: {:.3f} '.format(i, accuracy[i])

    return acc_str


# @jit(parallel=True)
def count_size(file):
    return os.stat(file).st_size


@jit
def chunk_split(data_list, splits):
    avg = len(data_list) / float(splits)
    out = []
    last = 0.0

    while last < len(data_list):
        out.append(data_list[int(last):int(last + avg)])
        last += avg

    return out


