#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 19/08/2019 11:02 $
# by : scchwagner $
# SVN : $
#

# --- imports -----------------------------------------------------------------
import torch
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F

from network.wrappers.NetworkBase import NetworkBase


class LeNet(NetworkBase):
    def __init__(self, network_type, loss, accuracy, lr, framework, training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2):
        """
        LeNet Convolutional Neural Network constructor
        :param loss:        used loss function
        :param lr:          learning rate
        :param training:    is training True/False
        :param num_filters: number of filters
        :param optimizer:   used optimizer
        :param nonlin:      used nonliniearity
        :param num_classes: number of classes/labels
        """
        super().__init__(network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes)
        self.weights, self.biases, self.nets = [], [], []


    def build_net(self, X):
        """
        Build the LeNet Convolutional Neural Network
        :param X:   input tensor
        :return:    network
        """

        with tf.name_scope('s_conv_1'):
            conv_1_1, batch_1_1, activ_1_1 = self._conv_bn_layer_tf(X, n_filters=6, filter_size=5,
                                                                 is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                 name_postfix='1_1')
            pooling_1 = tf.layers.average_pooling2d(activ_1_1, pool_size=2, strides=2, padding='valid', name='pooling_1')
            self.nets.extend([conv_1_1])

        with tf.name_scope('s_conv_2'):
            conv_2_1, batch_2_1, activ_2_1 = self._conv_bn_layer_tf(pooling_1, n_filters=16,
                                                                 filter_size=5, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, name_postfix='2_1')
            pooling_2 = tf.layers.average_pooling2d(activ_2_1, pool_size=2, strides=3, padding='valid', name='pooling_2')
            flat_1 = tf.layers.flatten(pooling_2, name='flatten')
            self.nets.extend([conv_2_1])

        with tf.name_scope('fl1'):
            fl_1 = tf.layers.dense(flat_1, units=120, activation='relu', name='fl_1')

        with tf.name_scope('fl2'):
            fl_2 = tf.layers.dense(fl_1, units=84, activation='relu', name='fl_2')

        with tf.name_scope('s_outputs'):
            output_p = tf.layers.dense(fl_2, units=self.num_classes, activation='softmax', name='output')

        return output_p


class LeNet_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes)
        nn.Module.__init__(self)

        # build net
        # Layer 1: conv and average pooling
        self.conv_1_1 = self._conv_bn_layer_pt(1, n_out=6, filter_size=5, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Layer 2: conv, avg pooling, faltten
        self.conv_2_1 = self._conv_bn_layer_pt(6, 16, filter_size=5, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Layer 3 and 4: FC
        self.fc_1 = nn.Linear(400, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, self.num_classes)

    def forward(self, X):
        x = self.conv_1_1(X)
        x = self.pool_1(x)
        x = self.conv_2_1(x)
        x = self.pool_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.out(x)

        return x
