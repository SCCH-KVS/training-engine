#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 11/07/2017 08:10 $
# by : shepeleva $
# SVN : $
#

# --- imports -----------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn
import tensorflow as tf
import numpy as np



def margin_tf(y_pred,y_true, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.

    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.

    Args:
      y_true: tensor, one hot encoding of ground truth.
      y_pred: tensor, model predictions in range [0, 1]
      margin: scalar, the margin after subtracting 0.5 from raw_logits.
      downweight: scalar, the factor for negative cost.

    Returns:
      A tensor with cost for each data point of shape [batch_size].
    """
    logits = y_pred - 0.5
    positive_cost = y_true * tf.cast(tf.less(logits, margin),
                                   tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - y_true) * tf.cast(
      tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return tf.reduce_mean(0.5 * positive_cost + downweight * 0.5 * negative_cost)


def IoU_tf(y_pred, y_true):
    """Returns a (approx) IOU score
    intersection = y_pred.flatten() * y_true.flatten()
    Then, IOU =  intersection / (y_pred.sum() + y_true.sum() - intersection)
    Args:
    :param y_pred: predicted labels (4-D array): (N, H, W, 1)
    :param y_true: groundtruth labels (4-D array): (N, H, W, 1)
    :return
    float: IOU score
    """
    threshold = 0.5
    axis = (0, 1, 2, 3)
    smooth = 1e-5
    pre = tf.cast(y_pred > threshold, dtype=tf.float32)
    truth = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou

    # b_iou = []
    # for i in range(b_s):
    #     lbl_iou = []
    #     for k in range(int(y_pred.shape[-1])):
    #         logits = tf.slice(y_pred, (i, 0, 0, k), (1, -1, -1, 1))
    #         trn_labels = tf.slice(y_true, (i, 0, 0, k), (1, -1, -1, 1))
    #         logits = tf.reshape(logits, [-1])
    #         trn_labels = tf.reshape(trn_labels, [-1])
    #         inter = tf.reduce_sum(tf.multiply(logits, trn_labels))
    #         denom = tf.reduce_sum(tf.subtract(tf.add(logits, trn_labels), tf.multiply(logits, trn_labels)))
    #         lbl_iou.append(tf.div(inter, denom))
    #     b_iou.append(lbl_iou)
    # return tf.div(tf.reduce_sum(b_iou, axis=0), b_s)


def dice_jaccard_tf(y_pred, y_true):
    """Returns a (approx) dice score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, dice = 2 * intersection / (y_pred.sum() + y_true.sum())
    :param y_pred: predicted labels (4-D array): (N, H, W, 1)
    :param y_true: groundtruth labels (4-D array): (N, H, W, 1)
    :return
        float: dice score
    """
    smooth = 1e-5
    inse = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2, 3))
    l = tf.reduce_sum(y_true * y_true, axis=(0, 1, 2, 3))
    r = tf.reduce_sum(y_pred * y_pred, axis=(0, 1, 2, 3))
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def dice_sorensen_tf(y_pred, y_true):
    smooth = 1e-5
    inse = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2, 3))
    l = tf.reduce_sum(y_true, axis=(0, 1, 2, 3))
    r = tf.reduce_sum(y_pred, axis=(0, 1, 2, 3))
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def softmax_tf(y_pred, y_true, epsilon=1e-10):
    """
    Computes cross entropy with included softmax - DO NOT provide outputs from softmax layers to this function!

    For brevity, let `x = output`, `z = target`.  The binary cross entropy loss is
    loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return
        cross entropy with included softmax
    """
    print(y_true)
    print(y_pred)

    # classification
    if len(y_true.shape) <= 2:
        # labels onehot
        if int(y_true.get_shape()[1]) > 1:
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        # labels encoded as integers
        else:
            return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y_true, logits=y_pred))
    # segmentation
    else:
        # print(y_true)
        # print(y_pred)
        y_true_flat = tf.reshape(y_true, [-1, int(y_true.shape[3])])
        y_pred_flat = tf.reshape(y_pred, [-1, int(y_true.shape[3])])
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_true_flat, logits=y_pred_flat))


def sigmoid_tf(y_pred, y_true):
    """
    Computes Sigmoid cross entropy
    :param y_pred:
    :param y_true:
    :return:
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))


def hinge_tf(y_pred, y_true):
    """
    Computes Hinge Loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Hinge Loss
    """
    return tf.losses.hinge_loss(labels=y_true, logits=y_pred)


def mse_tf(y_pred, y_true):
    """
    Computes Sum-of-Squares loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Sum-of-Squares loss
    """
    # y_p_s = tf.nn.sigmoid(y_pred)
    # return tf.losses.mean_squared_error(labels=y_true, predictions=y_p_s) + tf.losses.get_regularization_loss()

    return tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)


def mse_loss_tf(y_pred, y_true):
    """
    Computes Sum-of-Squares loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Sum-of-Squares loss
    """

    # init_shape = y_true.get_shape().as_list()

    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    aa_list = ((y_true_flat - y_pred_flat) * (y_true_flat - y_pred_flat))/2

    aa_reshaped = tf.reshape(aa_list, tf.shape(y_true))

    lst = tf.sqrt(tf.cast(tf.reduce_sum(aa_reshaped, axis=1), tf.float16))




    # import numpy as np
    # lst = []
    # dim = y_true.get_shape().as_list()
    #
    # for item in range(0,dim[0]):
    #     aa_list = [(i-j)^2 for i, j in zip(y_true[item], y_pred[item])]
    #     lst.append(np.sqrt(sum(aa_list)/2))

    return tf.reduce_mean(tf.cast(lst, dtype=tf.float32))
    # return tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)


def percentage_tf(y_pred, y_true):
    """
    Computes percentage of correct predictions
    :param y_pred:
    :param y_true:
    :return:
    """
    print(y_pred)
    print(y_true)
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))


# pytorch metrics

def margin_pt(y_pred,y_true, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.

    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.

    Args:
      y_true: tensor, one hot encoding of ground truth.
      y_pred: tensor, model predictions in range [0, 1]
      margin: scalar, the margin after subtracting 0.5 from raw_logits.
      downweight: scalar, the factor for negative cost.

    Returns:
      A tensor with cost for each data point of shape [batch_size].
    """
    true = y_true.max(dim=1)[1]
    return nn.SoftMarginLoss()(y_pred, true)


def IoU_pt(y_pred, y_true):
    """Returns a (approx) IOU score
    intersection = y_pred.flatten() * y_true.flatten()
    Then, IOU =  intersection / (y_pred.sum() + y_true.sum() - intersection)
    Args:
    :param y_pred: predicted labels (4-D array): (N, H, W, 1)
    :param y_true: groundtruth labels (4-D array): (N, H, W, 1)
    :return
    float: IOU score
    """
    smooth = 1.
    y_pred_sig = F.sigmoid(y_pred)
    num = y_true.size(0)  # Number of batches
    x = y_pred_sig.view(num, -1).float()  # Flatten
    y = y_true.view(num, -1).float()
    intersection = torch.sum(x * y)
    score = (intersection + smooth) / (torch.sum(x) + torch.sum(y) - intersection + smooth)
    out = torch.sum(score)
    print("iou {}".format(out))
    return out



def dice_loss(y_pred, y_true, eps=1e-7):
    return DiceLoss_pt()(y_pred, y_true)

class DiceLoss_pt(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_pt, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 1.
        y_pred_sig = F.sigmoid(y_pred)
        num = y_true.size(0)  # Number of batches
        x = y_pred_sig.view(num, -1).float()  # Flatten
        y = y_true.view(num, -1).float()
        intersection = torch.sum(x * y)
        score = (2. * intersection + smooth) / (torch.sum(x) + torch.sum(y) + smooth)
        out = 1 - torch.sum(score) / num
        return out

def softmax_pt(y_pred, y_true, epsilon=1e-10):
    """
    Computes cross entropy with included softmax - DO NOT provide outputs from softmax layers to this function!

    For brevity, let `x = output`, `z = target`.  The binary cross entropy loss is
    loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return
        cross entropy with included softmax
    """
    true = y_true.max(dim=1)[1]
    return nn.CrossEntropyLoss()(y_pred, true)


def sigmoid_pt(y_pred, y_true):
    """
    Computes Sigmoid cross entropy
    :param y_pred:
    :param y_true:
    :return:
    """
    true = y_true.max(dim=1)[1]
    return nn.BCEWithLogitsLoss()(y_pred, true)


def hinge_pt(y_pred, y_true):
    """
    Computes Hinge Loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Hinge Loss
    """
    raise NotImplementedError


def mse_pt(y_pred, y_true):
    """
    Computes Sum-of-Squares loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Sum-of-Squares loss
    """
    raise NotImplementedError


def mse_loss_pt(y_pred, y_true):
    """
    Computes Sum-of-Squares loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Sum-of-Squares loss
    """

    raise NotImplementedError


def percentage_pt(y_pred, y_true):
    """
    Computes percentage of correct predictions
    :param y_pred:
    :param y_true:
    :return:
    """
    y_pred_soft = y_pred.exp() / (y_pred.exp().sum(-1)).unsqueeze(-1)

    perc = (y_pred_soft.max(dim=1)[1] == y_true.max(dim=1)[1]).sum()
    return perc

