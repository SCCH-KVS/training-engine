#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 12/07/2017 13:57 $
# by : shepeleva $
# SVN : $
#

# --- imports -----------------------------------------------------------------

from utils.metric import *
import numpy as np
import torch.nn as nn
import torch.optim as optim

class NetworkBase:
    """

    """

    def __init__(self, network_type, loss, accuracy, lr, training, framework, optimizer=None, nonlin=None, upconv=None, num_filters=None, num_classes=None,
                 dropout=None, num_steps=None, trainable_layers=None):
        """
        Construtcor of NetworkBase Class
        :param loss:        loss function
        :param lr:          learning rate
        :param training:    is training True/False
        :param optimizer:   optimizer
        :param nonlin:      nonliniearity
        :param upconv:      upconvolution method
        :param num_filters: number of filters
        :param num_classes: number of classes/labels
        :param dropout:     dropout ratio
        """
        self.loss_f = self._pick_loss_func(loss, framework)
        # self.accuracy_f_list = self._pick_accuracy_function(accuracy)
        self.is_training = training
        self.learning_rate = lr
        self.optimizer = optimizer
        self.framework = framework
        self.network_type = network_type
        if optimizer:
            self.optimizer_f = self._pick_optimizer_func(optimizer, framework)
        if nonlin:
            self.nonlin_f = self._pick_nonlin_func(nonlin, framework)
        if upconv:
            self.upconv_f = self._pick_upconv_func(upconv, framework)
        if num_filters:
            self.num_filters = num_filters
        if num_classes:
            self.num_classes = num_classes
        if dropout:
            self.dropout = dropout
        if num_steps:
            self.num_steps = num_steps

        if trainable_layers:
            self.trainable_layers = trainable_layers
        else:
            self.trainable_layers = 'all'

        if isinstance(accuracy, list):
            self.accuracy_f_list = [self._pick_accuracy_function(acc, framework) for acc in accuracy]
        else:
            self.accuracy_f_list = self._pick_accuracy_function(accuracy, framework)

    def _loss_function(self, y_pred, y_true):
        """
        Returns the loss
        :param y_pred:  prediction
        :param y_true:  ground truth
        :return:        loss
        """
        self.loss = self.loss_f(y_pred=y_pred, y_true=y_true)
        return self.loss

    def _accuracy_function(self, y_pred, y_true, b_s, net_type, loss_type):
        """
        Returns the accuracy
        :param y_pred:
        :param y_true:
        :return:
        """

        if loss_type == 'sigmoid':
            y_pred = tf.nn.sigmoid(y_pred)
        if loss_type == 'softmax':
            y_pred = tf.nn.softmax(y_pred)

        if net_type == 'classification':
            if isinstance(self.accuracy_f_list, list) and (len(self.accuracy_f_list) >= 2):
                return [acc_fun(y_pred=y_pred, y_true=y_true, b_s=b_s) for acc_fun in self.accuracy_f_list]
            else:
                return self.accuracy_f_list(y_pred=y_pred, y_true=y_true)
        elif net_type == 'segmentation':
            if isinstance(self.accuracy_f_list, list) and (len(self.accuracy_f_list) >= 2):
                return [acc_fun(y_pred=y_pred, y_true=y_true, b_s=b_s) for acc_fun in self.accuracy_f_list]
            else:
                return self.accuracy_f_list(y_pred=y_pred, y_true=y_true)
        else:
            raise ValueError('Unexpected network task %s' % net_type)

    def _optimizer_function(self, global_step=None, net_param=None):
        """
        Return the optimizer function
        :param global_step: current global step
        :return:            optimizer
        """
        if global_step:
            if self.trainable_layers == 'all':
                return self.optimizer_f(self.learning_rate).minimize(self.loss, global_step=global_step)
            else:
                first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     self.trainable_layers)
                return self.optimizer_f(self.learning_rate).minimize(self.loss, var_list=first_train_vars,
                                                                     global_step=global_step)
        else:
            return self.optimizer_f(net_param, self.learning_rate)

    @staticmethod
    def _pick_nonlin_func(key, framework):
        """
        Select a nonlinearity/activation function
        :param key: nonliniearity identifier
        :return:    nonliniearity/activation function
        """
        if framework == "tensorflow":
            if key == 'elu':
                return tf.nn.elu
            elif key == 'relu':
                return tf.nn.relu
            elif key == 'lrelu':
                return tf.nn.leaky_relu
            elif key == 'tanh':
                return tf.nn.tanh
            else:
                raise ValueError('Unexpected nonlinearity function %s' % key)
        elif framework == "pytorch":
            if key == 'elu':
                return nn.ELU
            elif key == 'relu':
                return nn.ReLU
            elif key == 'lrelu':
                return nn.LeakyReLU
            elif key == 'tanh':
                return nn.Tanh
            else:
                raise ValueError('Unexpected nonlinearity function %s' % key)

    @staticmethod
    def _pick_upconv_func(key, framework):
        """
        Select either deconvolution or upsampling
        :param key: identifier
        :return:    upconv or upsampling
        """
        if framework == "tensorflow":
            if key == 'upconv':
                return tf.layers.conv2d_transpose
            if key == 'upsampling':

                def upconv2d(input_, output_shape, filters, kernel_size=4, strides=(1, 1), name="upconv2d"):
                    """
                    Pure tensorflow 2d upconvolution/upsampling layer using tf.image.resize_images
                    :param input_:
                    :param output_shape:
                    :param filters:
                    :param kernel_size:
                    :param strides:
                    :param name:
                    :return:
                    """
                    with tf.variable_scope(name):
                        resized = tf.image.resize_images(input_, output_shape[1:3])
                        upconv = tf.layers.conv2d(resized, filters=filters, kernel_size=kernel_size, strides=strides,
                                                  padding='same', name='conv')
                        # upconv = tf.nn.conv2d(resize, w, strides=[1, d_h, d_w, 1], padding='SAME')

                        return upconv

                return upconv2d
            else:
                raise ValueError('Unexpected upconvolution function %s' % key)
        elif framework == "pytorch":
            if key == 'upconv':
                return nn.ConvTranspose2d
            else:
                raise ValueError('Unexpected upconvolution function %s' % key)

    @staticmethod
    def _pick_loss_func(key, framework):
        """
        Select loss function
        :param key: loss function identifier
        :return:    loss function
        """
        if framework == "tensorflow":
            if key == 'softmax':
                return softmax_tf
            elif key == 'sigmoid':
                return sigmoid_tf
            elif key == 'margin':
                return margin_tf
            elif key == 'mse':
                return mse_tf
            elif key == 'mse_loss':
                return mse_loss_tf
            elif key == 'cross-entropy':
                return cross_entropy_tf
            else:
                raise ValueError('Unexpected metric function %s' % key)
        elif framework == "pytorch":
            if key == 'softmax':
                return softmax_pt
            elif key == 'sigmoid':
                return sigmoid_pt
            elif key == 'margin':
                return margin_pt
            elif key == 'mse':
                return mse_pt
            elif key == 'mse_loss':
                return mse_loss_pt
            elif key == 'dice_jaccard':
                return dice_loss
            elif key == 'cross-entropy':
                return softmax_pt
            else:
                raise ValueError('Unexpected metric function %s' % key)

    @staticmethod
    def _pick_accuracy_function(key, framework):
        if framework == "tensorflow":
            if key == 'IoU':
                return IoU_tf
            elif key == 'dice_sorensen':
                return dice_sorensen_tf
            elif key == 'dice_jaccard':
                return dice_jaccard_tf
            elif key == 'mse':
                return mse_tf
            elif key == 'hinge':
                return hinge_tf
            elif key == 'percent':
                return percentage_tf
            else:
                raise ValueError('Unexpected metric function %s' % key)
        elif framework == "pytorch":
            if key == 'IoU':
                return IoU_pt
            elif key == 'mse':
                return mse_pt
            elif key == 'hinge':
                return hinge_pt
            elif key == 'percent':
                return percentage_pt
            else:
                raise ValueError('Unexpected metric function %s' % key)


    @staticmethod
    def _pick_optimizer_func(key, framework):
        if framework == "tensorflow":
            if key == 'adam':
                return tf.train.AdamOptimizer
            if key == 'amsgrad':
                from network.third_parties.amsgrad.amsgrad import AMSGrad
                return AMSGrad
            if key == 'momentum':
                return tf.train.MomentumOptimizer
            if key == 'gradient':
                return tf.train.GradientDescentOptimizer
            if key == 'proximalgrad':
                return tf.train.ProximalGradientDescentOptimizer
            else:
                raise ValueError('Unexpected optimizer function %s' % key)
        elif framework == "pytorch":
            if key == 'adam':
                return optim.Adam
            if key == 'gradient':
                return optim.SGD
            else:
                raise ValueError('Unexpected optimizer function %s' % key)


    @staticmethod
    def _conv_bn_layer_tf(input_layer, n_filters, filter_scale=1, filter_size=3, is_training=True, nonlin_f=None,
                       padding='same', name='s_conv_bn', name_postfix='1_1', strides=None):
        """
        Convolution layer with batch normalization
        :param input_layer:     input layer
        :param n_filters:       number of filters
        :param filter_scale:    filter scale -> n_filters = n_filters * filter_scale
        :param filter_size:     filter size
        :param is_training:     training True/False
        :param nonlin_f:        nonlinearity/activation function, if None linear/no activation is used
        :param padding:         padding: valid or same
        :param name:            layer name
        :param name_postfix:    layer name postfix
        :return: conv, batch_norm, nonlin - convolution, batch normalization and nonlinearity/activation layer
        """
        with tf.name_scope(name + name_postfix):
            conv = tf.layers.conv2d(input_layer, filters=filter_scale * n_filters, kernel_size=filter_size,
                                    activation=None, padding=padding,
                                    name='conv_' + name_postfix, strides=strides)
            batch_norm = tf.layers.batch_normalization(conv, training=is_training, fused=False, name='batch_' + name_postfix)
                # weights, biases = NetworkBase.weights_and_biases(tf.shape(batch_norm), tf.shape(batch_norm)[-1])
                # nonlin = nonlin_f(tf.matmul(batch_norm, weights) + biases, name='activation_' + name_postfix)
            nonlin = nonlin_f(conv, name='activation_' + name_postfix)
        return conv, batch_norm, nonlin

    @staticmethod
    def _conv_layer_tf(input_layer, mu, sigma, padding, shape, filters, strides):
        conv1_w = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(filters))
        conv1 = tf.nn.conv2d(input_layer, conv1_w, strides=strides, padding=padding) + conv1_b
        conv1 = tf.nn.relu(conv1)
        return conv1

    @staticmethod
    def _fully_connected_layer_tf(input_layer, input, output, mu, sigma):
        fc1_w = tf.Variable(tf.truncated_normal(shape=(input, output), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(output))
        fc1 = tf.matmul(input_layer, fc1_w) + fc1_b
        return fc1

    @staticmethod
    def _conv_bn_layer_pt(n_in, n_out, filter_size=3, stride=1, is_training=True, nonlin_f=None,
                       padding=1, name_postfix='1_1'):
        m = nn.Sequential(
            nn.Conv2d(n_in, n_out, filter_size, stride, padding),
            nn.BatchNorm2d(n_out),
            nonlin_f()
        )
        return m
        # conv = nn.Conv2d(n_filters * filter_scale, input_layer, kernel_size=filter_size, stride=2)
        # batch_norm = nn.BatchNorm2d(conv)
        # nonlin = nonlin_f(batch_norm)
        # return conv, batch_norm, nonlin

    @staticmethod
    def _conv_layer_pt(n_in, n_out, filter_size=3, stride=1, is_training=True, nonlin_f=None,
                       padding=1, name_postfix='1_1'):
        m = nn.Sequential(
            nn.Conv2d(n_in, n_out, filter_size, stride, padding)
        )
        return m
        # conv = nn.Conv2d(n_filters * filter_scale, input_layer, kernel_size=filter_size, stride=2)
        # return conv


    @staticmethod
    def _sep_conv_bn_layer(input_layer, n_filters, filter_scale=1, filter_size=3, is_training=True, nonlin_f=None,
                       padding='same', name='s_conv_bn', name_postfix='1_1'):
        """
        Convolution layer with batch normalization
        :param input_layer:     input layer
        :param n_filters:       number of filters
        :param filter_scale:    filter scale -> n_filters = n_filters * filter_scale
        :param filter_size:     filter size
        :param is_training:     training True/False
        :param nonlin_f:        nonlinearity/activation function, if None linear/no activation is used
        :param padding:         padding: valid or same
        :param name:            layer name
        :param name_postfix:    layer name postfix
        :return: conv, batch_norm, nonlin - convolution, batch normalization and nonlinearity/activation layer
        """
        with tf.name_scope(name + name_postfix):
            conv = tf.layers.separable_conv2d(input_layer, filters=filter_scale * n_filters, kernel_size=filter_size,
                                    activation=None, padding=padding, depth_multiplier=3, name='conv_' + name_postfix)
            batch_norm = tf.layers.batch_normalization(conv, training=is_training, fused=False, name='batch_' + name_postfix)
            if nonlin_f:
                nonlin = nonlin_f(batch_norm, name='activation_' + name_postfix)
            else:
                nonlin = None
        return conv, batch_norm, nonlin

    @staticmethod
    def _conv_nonlin_layer(input_layer, n_filters, filter_scale=1, filter_size=3, is_training=True, nonlin_f=None,
                       padding='same', name='s_conv_nonlin', name_postfix='1_1'):
        """
        Convolution layer with batch normalization
        :param input_layer:     input layer
        :param n_filters:       number of filters
        :param filter_scale:    filter scale -> n_filters = n_filters * filter_scale
        :param filter_size:     filter size
        :param is_training:     training True/False
        :param nonlin_f:        nonlinearity/activation function, if None linear/no activation is used
        :param padding:         padding: valid or same
        :param name:            layer name
        :param name_postfix:    layer name postfix
        :return: conv, batch_norm, nonlin - convolution, batch normalization and nonlinearity/activation layer
        """
        with tf.name_scope(name + name_postfix):
            conv = tf.layers.conv2d(input_layer, filters=filter_scale * n_filters, kernel_size=filter_size,
                                    activation=None, padding=padding, name='conv_' + name_postfix)
            if nonlin_f:
                nonlin = nonlin_f(conv, name='activation_' + name_postfix)
            else:
                nonlin = None
        return conv, nonlin

    @staticmethod
    def _pool_layer(net, pool_size, stride_size, type='max', padding='same', name='pooling'):
        """
        Pooling layer
        :param net:             network layer
        :param pool_size:       pooling size
        :param stride_size:     stride size
        :param type:            pooling type: max, average
        :param padding:         padding: valid or same
        :param name:            layer name
        :return:
        """
        pre_pool = net
        if type == 'max':
            net = tf.layers.max_pooling2d(net, pool_size, stride_size, padding=padding, name=name)
        elif type == 'avg':
            net = tf.layers.average_pooling2d(net, pool_size, stride_size, padding=padding, name=name)
        return pre_pool, net

    def return_accuracy(self, y_pred, y_true, b_s=None, net_type=None, loss_type=None):
        """
        Returns the prediction accuracy
        :param y_pred:  prediction
        :param y_true:  ground truth
        :return:        accuracy
        """
        return self._accuracy_function(y_pred=y_pred, y_true=y_true, b_s=b_s, net_type=net_type, loss_type=loss_type)

    def return_loss(self, y_pred, y_true):
        """
        Returns the loss
        :param y_pred:  prediction
        :param y_true:  ground truth
        :return:        loss
        """
        return self._loss_function(y_pred=y_pred, y_true=y_true)

    def return_optimizer(self, global_step=None, net_param=None):
        """
        Returns the optimizer function
        :param global_step: current global step
        :return:            optimizer
        """
        return self._optimizer_function(global_step, net_param)

    def return_nets(self):
        """
        Returns the network (only the convolutional layers)
        :return:    network
        """
        return self.nets

    @staticmethod
    def weights_and_biases(a, b):
        w = tf.Variable(tf.truncated_normal(shape=[a, b], stddev=np.sqrt(2 / a)))
        b = tf.Variable(tf.zeros([b]))

        return w, b

