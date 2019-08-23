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

import torch.nn as nn
import tensorflow as tf

from network.wrappers.NetworkBase import NetworkBase


class ResNet(NetworkBase):
    def __init__(self, network_type, loss, accuracy, lr, framework, training, trainable_layers=None, num_filters=64,
                 optimizer='adam', nonlin='elu', num_classes=2):
        """
        ResNet Convolutional Neural Network constructor
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
        Build the ResNet Convolutional Neural Network
        :param X:   input tensor
        :return:    network
        """

        # Stage 1
        with tf.name_scope('s_stage_1'):
            conv_1_1, batch_1_1, activ_1_1 = self._conv_bn_layer_tf(X, n_filters=self.num_filters, filter_size=7,
                                                                    is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f,
                                                                    name_postfix='1_1')
            pooling_1 = tf.layers.max_pooling2d(activ_1_1, pool_size=3, strides=2, padding='valid', name='pooling_1')
            self.nets.extend([conv_1_1])

        # Stage 2
        with tf.name_scope('s_stage_2'):
            conv_2_1 = self._convolutional_block(pooling_1, filters=[64, 64], i=21)
            conv_2_2 = self._identity_block(conv_2_1,  filters=[64, 64], i=22)
            conv_2_3 = self._identity_block(conv_2_2, filters=[64, 64], i=23)

        # Stage 3
        with tf.name_scope('s_stage_3'):
            conv_3_1 = self._convolutional_block(conv_2_3, filters=[128, 128], i=31)
            pooling_2 = tf.layers.max_pooling2d(conv_3_1, pool_size=3, strides=2, padding='valid', name='pooling_2')
            conv_3_2 = self._identity_block(pooling_2,  filters=[128, 128], i=32)
            conv_3_3 = self._identity_block(conv_3_2, filters=[128, 128], i=33)

        # Stage 4
        with tf.name_scope('s_stage_4'):
            conv_4_1 = self._convolutional_block(conv_3_3, filters=[256, 256], i=41)
            pooling_3 = tf.layers.max_pooling2d(conv_4_1, pool_size=3, strides=2, padding='valid', name='pooling_3')
            conv_4_2 = self._identity_block(pooling_3, filters=[256, 256], i=42)
            conv_4_3 = self._identity_block(conv_4_2, filters=[256, 256], i=43)

        # Stage 5
        with tf.name_scope('s_stage_5'):
            conv_5_1 = self._convolutional_block(conv_4_3, filters=[512, 512], i=51)
            pooling_4 = tf.layers.max_pooling2d(conv_5_1, pool_size=3, strides=2, padding='valid', name='pooling_4')
            conv_5_2 = self._identity_block(pooling_4, filters=[512, 512], i=52)
            conv_5_3 = self._identity_block(conv_5_2, filters=[512, 512], i=53)

        # Output Layer
        with tf.name_scope('s_outputs'):
            pooling_5 = tf.layers.average_pooling2d(conv_5_3, pool_size=2, strides=2, padding='same', name='pooling_5')
            flat = tf.layers.flatten(pooling_5, name='flatten')
            output_p = tf.layers.dense(flat, units=self.num_classes, activation='softmax', name='output')
        return output_p

    def _identity_block(self, X, filters, i):
        # Retrieve Filters
        F1, F2 = filters

        conv_2_1, batch_2_1, activ_2_1 = self._conv_bn_layer_tf(X, n_filters=F1, filter_size=3,
                                                                is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                name_postfix='1_1'+str(i))

        conv_2_2, batch_2_2, activ_2_2 = self._conv_bn_layer_tf(activ_2_1, n_filters=F2, filter_size=3,
                                                                is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                name_postfix='1_2'+str(i))

        self.nets.extend([conv_2_1, conv_2_2, conv_2_2])
        return activ_2_2

    def _convolutional_block(self, X, filters, i):
        # Retrieve Filters
        F1, F2 = filters

        # Save the input value
        x_shortcut = X

        ##### MAIN PATH #####
        conv_2_1, batch_2_1, activ_2_1 = self._conv_bn_layer_tf(X, n_filters=F1, filter_size=3,
                                                                is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                name_postfix='1_1'+str(i))

        conv_2_2, batch_2_2, activ_2_2 = self._conv_bn_layer_tf(activ_2_1, n_filters=F2, filter_size=3,
                                                                is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                name_postfix='1_2'+str(i))

        ##### SHORTCUT PATH ####
        conv_2_3, batch_2_3, activ_2_3 = self._conv_bn_layer_tf(x_shortcut, n_filters=F2, filter_size=3,
                                                                 is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f,
                                                                 name_postfix='1_3'+str(i))

        batch_2_2 += batch_2_3

        self.nets.extend([conv_2_1, conv_2_2, conv_2_3])
        return tf.nn.relu(batch_2_2)


class ResNet_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes)
        nn.Module.__init__(self)


        # Stage 1
        self.conv_1_1 = self._conv_bn_layer_pt(3, 64, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Stage 2
        self.layer1_2 = self._identityblock([64, 64], 1, 12)
        self.layer1_3 = self._identityblock([64, 64], 1, 13)
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Stage 3
        self.layer2_2 = self._identityblock([64, 128], 2, 22)
        self.layer2_3 = self._identityblock([128, 128], 2, 23)
        self.pool_3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Stage 4
        self.layer3_2 = self._identityblock([128, 128], 2, 32)
        self.layer3_3 = self._identityblock([128, 128], 2, 33)
        self.pool_4 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Stage 5
        self.layer4_2 = self._identityblock([128, 128], 2, 42)
        self.layer4_3 = self._identityblock([128, 128], 2, 43)

        # Output
        self.pool_5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out = nn.Linear(128, self.num_classes)

    def _convolutionalblock(self):
        return self

    def _identityblock(self, filters, stride, i):
        in_no, out_no = filters

        return self._conv_bn_layer_pt(in_no, out_no, filter_size=3, stride=stride, is_training=True,
                                                   nonlin_f=self.nonlin_f, padding=1,
                                                   name_postfix='1_1'+str(i))

    def forward(self, X):

        # Stage 1
        x = self.conv_1_1(X)
        x = self.pool_1(x)

        # Stage 2
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        x = self.pool_2(x)

        # Stage 3
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        #x = self.pool_3(x)

        # Stage 4
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        #x = self.pool_4(x)

        # Stage 5
        #x = self.layer4_2(x)
        #x = self.layer4_3(x)
        #x = self.pool_5(x)
        #print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x
