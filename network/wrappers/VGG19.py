#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 3/21/2019 8:32 AM $ 
# by : Shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------

import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F

from network.wrappers.NetworkBase import NetworkBase


class VGG19_tf(NetworkBase):
    def __init__(self, network_type, loss, accuracy, lr, framework, training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2, dropout=0.25):
        """
        Convolutional Neural Network constructor
        :param loss:        used loss function
        :param lr:          learning rate
        :param training:    is training True/False
        :param num_filters: number of filters
        :param optimizer:   used optimizer
        :param nonlin:      used nonliniearity
        :param num_classes: number of classes/labels
        :param dropout:     dropout ratio
        """
        super().__init__(network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr,
                         training=training,
                         trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                         num_classes=num_classes, dropout=dropout)
        self.weights, self.biases, self.nets = [], [], []

    def build_net(self, X):
        """
        Build the Convolutional Neural Network
        :param X:   input tensor
        :return:    network
        """

        with tf.name_scope('s_conv_1'):
            conv_1_1, batch_1_1, activ_1_1 = self._conv_bn_layer_tf(X, n_filters=self.num_filters, filter_size=3,
                                                                    is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f,
                                                                    name_postfix='1_1')
            conv_1_2, batch_1_2, activ_1_2 = self._conv_bn_layer_tf(activ_1_1, n_filters=self.num_filters,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='1_2')
            pooling_1 = tf.layers.max_pooling2d(activ_1_2, pool_size=2, strides=2, padding='same', name='pooling_1')
            self.nets.extend([conv_1_1, conv_1_2])

        with tf.name_scope('s_conv_2'):
            conv_2_1, batch_2_1, activ_2_1 = self._conv_bn_layer_tf(pooling_1, n_filters=self.num_filters, filter_scale=2,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='2_1')
            conv_2_2, batch_2_2, activ_2_2 = self._conv_bn_layer_tf(activ_2_1, n_filters=self.num_filters,
                                                                    filter_scale=2,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='2_2')
            pooling_2 = tf.layers.max_pooling2d(activ_2_2, pool_size=2, strides=2, padding='same', name='pooling_2')
            self.nets.extend([conv_2_1, conv_2_2])

        with tf.name_scope('s_conv_3'):
            conv_3_1, batch_3_1, activ_3_1 = self._conv_bn_layer_tf(pooling_2, n_filters=self.num_filters, filter_scale=4,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='3_1')
            conv_3_2, batch_3_2, activ_3_2 = self._conv_bn_layer_tf(activ_3_1, n_filters=self.num_filters,
                                                                    filter_scale=4,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='3_2')
            conv_3_3, batch_3_3, activ_3_3 = self._conv_bn_layer_tf(conv_3_2, n_filters=self.num_filters, filter_scale=4,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='3_3')
            conv_3_4, batch_3_2, activ_3_4 = self._conv_bn_layer_tf(conv_3_3, n_filters=self.num_filters,
                                                                    filter_scale=4,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='3_4')
            pooling_3 = tf.layers.max_pooling2d(conv_3_4, pool_size=3, strides=2, padding='same', name='pooling_3')

            self.nets.extend([conv_3_1, conv_3_2])

        with tf.name_scope('s_conv_4'):
            conv_4_1, batch_4_1, activ_4_1 = self._conv_bn_layer_tf(pooling_3, n_filters=self.num_filters, filter_scale=8,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='4_1')
            conv_4_2, batch_4_2, activ_4_2 = self._conv_bn_layer_tf(activ_4_1, n_filters=self.num_filters,
                                                                    filter_scale=8,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='4_2')
            conv_4_3, batch_4_3, activ_4_3 = self._conv_bn_layer_tf(conv_4_2, n_filters=self.num_filters, filter_scale=8,
                                                                  filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='4_3')
            conv_4_4, batch_4_2, activ_4_4 = self._conv_bn_layer_tf(conv_4_3, n_filters=self.num_filters,
                                                                    filter_scale=8,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='4_4')
            pooling_4 = tf.layers.max_pooling2d(conv_4_4, pool_size=3, strides=2, padding='same', name='pooling_4')

        with tf.name_scope('s_conv_5'):
            conv_5_1, batch_5_1, activ_5_1 = self._conv_bn_layer_tf(pooling_4, n_filters=self.num_filters, filter_scale=8,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='5_1')
            conv_5_2, batch_5_2, activ_5_2 = self._conv_bn_layer_tf(activ_5_1, n_filters=self.num_filters,
                                                                    filter_scale=8,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='5_2')
            conv_5_3, batch_5_3, activ_5_3 = self._conv_bn_layer_tf(conv_5_2, n_filters=self.num_filters, filter_scale=8,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='5_3')
            conv_5_4, batch_5_2, activ_5_4 = self._conv_bn_layer_tf(conv_5_3, n_filters=self.num_filters,
                                                                    filter_scale=8,
                                                                    filter_size=3, is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f, name_postfix='5_4')
            pooling_5 = tf.layers.max_pooling2d(conv_5_4, pool_size=3, strides=2, padding='same', name='pooling_5')

        with tf.name_scope('s_outputs'):
            flat = tf.layers.flatten(pooling_5, name='flatten')
            drop_5 = tf.layers.dropout(flat, rate=self.dropout, name='drop_5')
            fc_1 = tf.layers.dense(drop_5, units=512, activation=self.nonlin_f, name='fc_1')
            drop_6 = tf.layers.dropout(fc_1, rate=self.dropout, name='drop_6')
            fc_2 = tf.layers.dense(drop_6, units=512, activation=self.nonlin_f, name='fc_2')
            # drop_6 = tf.layers.dropout(fc_2, rate=self.dropout, name='drop_6')
            output_p = tf.layers.dense(fc_2, units=self.num_classes, activation=None, name='output')
        return output_p


class VGG19_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2, dropout=0.25):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes, dropout=dropout)
        nn.Module.__init__(self)

        self.conv_1_1 = self._conv_bn_layer_pt(3, self.num_filters, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_1_2 = self._conv_bn_layer_pt(num_filters, self.num_filters, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2_1 = self._conv_bn_layer_pt(num_filters, self.num_filters * 2, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_2_2 = self._conv_bn_layer_pt(num_filters * 2, self.num_filters * 2, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3_1 = self._conv_bn_layer_pt(num_filters * 2, self.num_filters * 4, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_3_2 = self._conv_bn_layer_pt(num_filters * 4, self.num_filters * 4, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_3_3 = self._conv_bn_layer_pt(num_filters * 4, self.num_filters * 4, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_3_4 = self._conv_bn_layer_pt(num_filters * 4, self.num_filters * 4, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_4_1 = self._conv_bn_layer_pt(num_filters * 4, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_4_2 = self._conv_bn_layer_pt(num_filters * 8, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_4_3 = self._conv_bn_layer_pt(num_filters * 8, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_4_4 = self._conv_bn_layer_pt(num_filters * 8, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_5_1 = self._conv_bn_layer_pt(num_filters * 8, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_5_2 = self._conv_bn_layer_pt(num_filters * 8, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_5_3 = self._conv_bn_layer_pt(num_filters * 8, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.conv_5_4 = self._conv_bn_layer_pt(num_filters * 8, self.num_filters * 8, filter_size=3, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding=1, name_postfix='1_1')
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_1 = nn.Dropout2d(p=0.5)
        self.fc_1 = nn.Linear(512, 512)
        self.n_1 = self.nonlin_f()
        self.drop_2 = nn.Dropout2d(p=0.5)
        self.fc_2 = nn.Linear(512, 512)
        self.n_2 = self.nonlin_f()
        self.drop_3 = nn.Dropout2d(p=0.5)
        self.out = nn.Linear(512, self.num_classes)

    def forward(self, X):
        x = self.conv_1_1(X)
        x = self.conv_1_2(x)
        x = self.pool_1(x)

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.pool_2(x)

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.conv_3_4(x)
        x = self.pool_3(x)

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.conv_4_4(x)
        x = self.pool_4(x)

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.conv_5_4(x)
        x = self.pool_5(x)

        x = x.view(x.size(0), -1)
        x = self.drop_1(x)
        x = self.fc_1(x)
        x = self.n_1(x)
        x = self.drop_2(x)
        x = self.fc_2(x)
        x = self.n_2(x)
        # x = self.drop_3(x)
        x = self.out(x)

        return x