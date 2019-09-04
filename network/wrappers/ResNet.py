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
                 optimizer='adam', nonlin='relu', num_classes=2):
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
            conv = tf.layers.conv2d(X, filters=self.num_filters, kernel_size=3, name='conv_1_1', padding='same')
            batch_norm = tf.layers.batch_normalization(conv, training=self.is_training, fused=False,
                                                       name='batch_1_1')
            nonlin = self.nonlin_f(batch_norm, name='activation_1_1')

        # Stage 2
        with tf.name_scope('s_stage_2'):
            conv_2_1 = self._basic_block(nonlin,  filters=64, i=21, strides=1)
            conv_2_2 = self._basic_block(conv_2_1, filters=64, i=22, strides=1)

        # Stage 3
        with tf.name_scope('s_stage_3'):
            conv_3_1 = self._basic_block(conv_2_2,  filters=128, i=31, strides=2)
            conv_3_2 = self._basic_block(conv_3_1, filters=128, i=32, strides=1)

        # Stage 4
        with tf.name_scope('s_stage_4'):
            conv_4_1 = self._basic_block(conv_3_2, filters=256, i=41, strides=2)
            conv_4_2 = self._basic_block(conv_4_1, filters=256, i=42, strides=1)

        # Stage 5
        with tf.name_scope('s_stage_5'):
            conv_5_1 = self._basic_block(conv_4_2, filters=512, i=51, strides=2)
            conv_5_2 = self._basic_block(conv_5_1, filters=512, i=52, strides=1)

        # Output Layer
        with tf.name_scope('s_outputs'):
            pooling_1 = tf.layers.average_pooling2d(conv_5_2, pool_size=2, strides=3, padding='valid', name='pooling_1')
            flat = tf.layers.flatten(pooling_1, name='flatten')
            output_p = tf.layers.dense(flat, units=self.num_classes, name='output', activation='softmax')
        return output_p

    def _basic_block(self, X, filters, i, strides):
        # Retrieve Filters
        conv_x_1 = tf.layers.conv2d(X, filters=filters, kernel_size=1, padding='same', name='conv_10_1_' + str(i),
                                    strides=strides)
        batch_norm_x_1 = tf.layers.batch_normalization(conv_x_1, training=self.is_training, fused=False,
                                                       name='batch_10_1_' + str(i))
        nonlin = self.nonlin_f(batch_norm_x_1, name='activation_10_1_' + str(i))
        conv_x_2 = tf.layers.conv2d(nonlin, filters=filters, kernel_size=3, padding='same', name='conv_10_2_' + str(i),
                                    strides=1)
        batch_norm_x_2 = tf.layers.batch_normalization(conv_x_2, training=self.is_training, fused=False,
                                                       name='batch_10_2_' + str(i))

        shortcut = tf.layers.Layer()
        if strides != 1 or X != filters:
            shortcut = tf.layers.conv2d(X, filters=filters, kernel_size=1, padding='valid', name='conv_10_3_' + str(i),
                                        strides=strides)
            shortcut = tf.layers.batch_normalization(shortcut, training=self.is_training, fused=False,
                                                     name='batch_10_3_' + str(i))
        output = batch_norm_x_2+shortcut
        output = self.nonlin_f(output, name='activation_10_2_' + str(i))

        return output


class ResNet_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=64,
                 optimizer='adam', nonlin='relu', num_classes=2):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes)
        nn.Module.__init__(self)
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(BasicBlock, 64, 2, 1)
        self.conv3_x = self._make_layer(BasicBlock, 128, 2, 2)
        self.conv4_x = self._make_layer(BasicBlock, 256, 2, 2)
        self.conv5_x = self._make_layer(BasicBlock, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, X):
        output = self.conv1(X)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
