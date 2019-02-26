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
import cv2
import time
import cython
from scipy.misc import imresize
# from skimage.transform import resize
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


@jit
def resize_images_gpu(x, img_size):
    # return cv2.resize(x, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    if x.shape == img_size:
        return x
    else:
        return imresize(x, (img_size[0], img_size[1]), interp='nearest')

@jit
def expand_dims(x, img_size):
    # return np.expand_dims(np.expand_dims(resize_images_gpu(x, img_size), axis=0), axis=-1)

    return np.expand_dims(np.expand_dims(x, axis=0), axis=-1)

@jit
def resize_images_cpu(x, img_size):
    return cv2.resize(x, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)


@cuda.jit
def preprocess_image(im_file, img_size):
    if img_size[2] == 3:
        img = cv2.imread(im_file, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    image = resize_images_gpu(img, img_size)
    return image

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


# def create_ckpt_data(file_path):
#     if not os.path.exists(file_path):
#         # create directory
#         os.makedirs(file_path)
#
#     return "{}/model.ckpt".format(file_path)