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
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=64,
                 optimizer='adam', nonlin='elu', num_classes=2):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes)
        nn.Module.__init__(self)
        self.in_channels = 64

        self.conv = self._conv_layer_pt(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = self.nonlin_f()
        self.layer1 = self._make_layer(ResidualBlock, 64, 2)
        self.avg_pool_1 = nn.AvgPool2d(2)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2)
        self.avg_pool_2 = nn.AvgPool2d(2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2)
        self.avg_pool_3 = nn.AvgPool2d(2)
        self.fc = nn.Linear(4096, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(self._conv_layer_pt(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        x = self.conv(X)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.avg_pool_1(x)
        x = self.layer2(x)
        x = self.avg_pool_2(x)
        x = self.layer3(x)
        x = self.avg_pool_3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, X):
        residual = X
        x = self.conv1(X)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x
