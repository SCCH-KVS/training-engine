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


        # Identity Block
        # convolutional block


        # batch normalization

        # Stage 1
        with tf.name_scope('s_stage_1'):
            conv_1_1, batch_1_1, activ_1_1 = self._conv_bn_layer_tf(X, n_filters=self.num_filters, filter_size=3,
                                                                    is_training=self.is_training,
                                                                    nonlin_f=self.nonlin_f,
                                                                    name_postfix='1_1')
            # btach nomr
            batch_norm_1 = tf.layers.batch_normalization(conv_1_1, axis=3)
            pooling_1 = tf.layers.max_pooling2d(batch_norm_1, pool_size=3, strides=2, padding='valid', name='pooling_1')
            self.nets.extend([conv_1_1])

        # Stage 2
        with tf.name_scope('s_stage_2'):
            conv_2 = self._convolutional_block(pooling_1, filters=[64, 64, 256], i=2)

        # Stage 3
        with tf.name_scope('s_stage_3'):
            conv_3 = self._convolutional_block(conv_2, filters=[128, 128, 512], i=3)

        # Stage 4
        with tf.name_scope('s_stage_4'):
            conv_4 = self._convolutional_block(conv_3, filters=[256, 256, 1024], i=4)

        # Output Layer
        with tf.name_scope('s_outputs'):
            pooling_2 = tf.layers.average_pooling2d(conv_4, pool_size=2, strides=2, padding='same', name='pooling_2')
            flat = tf.layers.flatten(pooling_2, name='flatten')
            output_p = tf.layers.dense(flat, units=self.num_classes, activation='softmax', name='output')
        return output_p

    def _convolutional_block(self, X, filters, i):

        F1, F2, F3 = filters
        #x_shortcut = X

        conv_2_1, batch_2_1, activ_2_1 = self._conv_bn_layer_tf(X, n_filters=F1, filter_size=3,
                                                                is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                name_postfix='1_1'+str(i))
        batch_norm_1 = tf.layers.batch_normalization(conv_2_1, axis=3)

        conv_2_2, batch_2_2, activ_2_2 = self._conv_bn_layer_tf(batch_norm_1, n_filters=F2, filter_size=3,
                                                                is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                name_postfix='1_2'+str(i))
        batch_norm_2 = tf.layers.batch_normalization(conv_2_2, axis=3)

        conv_2_3, batch_2_3, activ_2_3 = self._conv_bn_layer_tf(batch_norm_2, n_filters=F3, filter_size=3,
                                                                is_training=self.is_training, nonlin_f=self.nonlin_f,
                                                                name_postfix='1_3'+str(i))
        batch_norm_3 = tf.layers.batch_normalization(conv_2_3, axis=3)

        #x_shortcut = self._conv_bn_layer_tf(x_shortcut, n_filters=F3, filter_size=3, is_training=self.is_training,
        #                                    nonlin_f=self.nonlin_f, name_postfix='1_4'+str(i))
        #x_shortcut = tf.layers.batch_normalization(x_shortcut, axis=3)


        self.nets.extend([conv_2_1, conv_2_2, conv_2_3])
        return batch_norm_3


class ResNet_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes)
        nn.Module.__init__(self)

    def forward(self, X):


        return X
