#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 12/07/2017 11:02 $
# by : shepeleva $
# SVN : $
#

# --- imports -----------------------------------------------------------------
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F

from network.wrappers.NetworkBase import NetworkBase


class ConvNet(NetworkBase):
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
        super().__init__(network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
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
                                                                 is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                 name_postfix='1_1')
            conv_1_2, batch_1_2, activ_1_2 = self._conv_bn_layer_tf(activ_1_1, n_filters=self.num_filters,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, name_postfix='1_2')
            pooling_1 = tf.layers.max_pooling2d(activ_1_2, pool_size=2, strides=2, padding='same', name='pooling_1')
            drop_1 = tf.layers.dropout(pooling_1, rate=self.dropout, name='drop_1')
            self.nets.extend([conv_1_1, conv_1_2])

        with tf.name_scope('s_conv_2'):
            conv_2_1, batch_2_1, activ_2_1 = self._conv_bn_layer_tf(drop_1, n_filters=self.num_filters, filter_scale=2,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, name_postfix='2_1')
            conv_2_2, batch_2_2, activ_2_2 = self._conv_bn_layer_tf(activ_2_1, n_filters=self.num_filters, filter_scale=2,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, name_postfix='2_2')
            pooling_2 = tf.layers.max_pooling2d(activ_2_2, pool_size=2, strides=2, padding='same', name='pooling_2')
            drop_2 = tf.layers.dropout(pooling_2, rate=self.dropout, name='drop_2')
            self.nets.extend([conv_2_1, conv_2_2])

        with tf.name_scope('s_conv_3'):
            conv_3_1, batch_3_1, activ_3_1 = self._conv_bn_layer_tf(drop_2, n_filters=self.num_filters, filter_scale=4,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, name_postfix='3_1')
            conv_3_2, batch_3_2, activ_3_2 = self._conv_bn_layer_tf(activ_3_1, n_filters=self.num_filters, filter_scale=4,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, name_postfix='3_2')
            pooling_3 = tf.layers.max_pooling2d(activ_3_2, pool_size=3, strides=2, padding='same', name='pooling_3')
            drop_3 = tf.layers.dropout(pooling_3, rate=self.dropout, name='drop_3')
            self.nets.extend([conv_3_1, conv_3_2])



        with tf.name_scope('s_outputs'):
            flat = tf.layers.flatten(drop_3, name='flatten')
            fc_1 = tf.layers.dense(flat, units=1024, activation=None, name='fc_1')
            drop_5 = tf.layers.dropout(fc_1, rate=self.dropout, name='drop_5')
            fc_2 = tf.layers.dense(drop_5, units=512, activation=None, name='fc_2')
            drop_6 = tf.layers.dropout(fc_2, rate=self.dropout, name='drop_6')
            output_p = tf.layers.dense(drop_6, units=self.num_classes, activation=None, name='output')
        return output_p


class ConvNet_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2, dropout=0.25):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes, dropout=dropout)
        nn.Module.__init__(self)

        self.conv_1_1 = self._conv_bn_layer_pt(1, self.num_filters, filter_size=5, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding='same', name_postfix='1_1')
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_1 = nn.Dropout2d(p=self.dropout)

        self.conv_2_1 = self._conv_bn_layer_pt(self.num_filters, self.num_filters*2, filter_size=5, stride=1, is_training=True,
                                               nonlin_f=self.nonlin_f, padding='same', name_postfix='1_1')
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_2 = nn.Dropout2d(p=self.dropout)


        # self.fc_1 = nn.Linear(1024, 512)
        # self.drop_4 = nn.Dropout2d(p=self.dropout)

        self.fc_2 = nn.Linear(512, 128)
        self.drop_5 = nn.Dropout2d(p=self.dropout)

        self.out = nn.Linear(128, self.num_classes)

    def forward(self, X):
        x = self.conv_1_1(X)
        # x = self.conv_1_2(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        x = self.conv_2_1(x)
        # x = self.conv_2_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        # x = self.conv_3_1(x)
        # x = self.conv_3_2(x)
        # x = self.pool_3(x)
        # x = self.drop_3(x)
        x = x.view(x.size(0), -1)
        # x = self.fc_1(x)
        # x = self.drop_4(x)
        x = self.fc_2(x)
        x = self.drop_5(x)
        x = self.out(x)

        return x
