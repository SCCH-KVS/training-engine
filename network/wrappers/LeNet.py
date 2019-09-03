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
from tensorflow.contrib.layers import flatten

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

        mu = 0
        sigma = 0.1
        #  Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1 = self._conv_layer_tf(X, mu, sigma, padding='VALID', shape=[5, 5, 1, 6], filters=6, strides=[1, 1, 1, 1])

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2 = self._conv_layer_tf(pool_1, mu, sigma, padding='VALID', shape=[5, 5, 6, 16], filters=16,
                                    strides=[1, 1, 1, 1])

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten. Input = 5x5x16. Output = 400.
        fc1 = flatten(pool_2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1 = self._fully_connected_layer_tf(fc1, 400, 120, mu, sigma)
        fc1 = tf.nn.relu(fc1)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2 = self._fully_connected_layer_tf(fc1, 120, 84, mu, sigma)
        fc2 = tf.nn.relu(fc2)

        # TLayer 5: Fully Connected. Input = 84. Output = 10.
        outputs = self._fully_connected_layer_tf(fc2, 84, 10, mu, sigma)
        return outputs


class LeNet_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes)
        nn.Module.__init__(self)

        # build net
        # Layer 1: conv and average pooling
        self.conv_1_1 = self._conv_bn_layer_pt(1, 6, filter_size=5, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=0, name_postfix='1_1')
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: conv, avg pooling, faltten
        self.conv_2_1 = self._conv_bn_layer_pt(6, 16, filter_size=5, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=0, name_postfix='1_1')
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

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
