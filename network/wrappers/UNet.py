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

import torch
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F

from network.wrappers.NetworkBase import NetworkBase


class UNet(NetworkBase):
    def __init__(self, network_type, loss, accuracy, lr, training, framework, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', upconv='upconv', num_classes=None):
        """

        :param is_training:
        """
        super().__init__(network_type, loss, accuracy, lr, training, framework, optimizer, nonlin, upconv, num_filters,
                         trainable_layers=trainable_layers, num_classes=num_classes)
        self.weights, self.biases, self.nets = [], [], []

    def build_net_tf(self, X):
        """

        :param X:        input tensor
        :return:
        """

        with tf.name_scope('encoder'):
            e_conv_1, e_batch_1, e_activ_1 = self._conv_bn_layer(X, n_filters=self.num_filters, filter_scale=1,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='e_conv_bn', name_postfix='_1')
            e_conv_2, e_batch_2, e_activ_2 = self._conv_bn_layer(e_activ_1, n_filters=self.num_filters, filter_scale=1,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f ,padding='same',
                                                                 name='e_conv_bn', name_postfix='_2')
            p1 = e_activ_2
            pool_1 = tf.layers.max_pooling2d(e_activ_2, pool_size=2, strides=2, padding='same', name='pooling_1')

            e_conv_3, e_batch_3, e_activ_3 = self._conv_bn_layer(pool_1, n_filters=self.num_filters, filter_scale=2,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='e_conv_bn', name_postfix='_3')
            e_conv_4, e_batch_4, e_activ_4 = self._conv_bn_layer(e_activ_3, n_filters=self.num_filters, filter_scale=2,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='e_conv_bn', name_postfix='_4')
            p2 = e_activ_4
            pool_2 = tf.layers.max_pooling2d(e_activ_4, pool_size=2, strides=2, padding='same', name='pooling_2')

            e_conv_5, e_batch_5, e_activ_5 = self._conv_bn_layer(pool_2, n_filters=self.num_filters, filter_scale=4,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='e_conv_bn', name_postfix='_5')
            e_conv_6, e_batch_6, e_activ_6 = self._conv_bn_layer(e_activ_5, n_filters=self.num_filters, filter_scale=4,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='e_conv_bn', name_postfix='_6')
            p3 = e_activ_6
            pool_3 = tf.layers.max_pooling2d(e_activ_6, pool_size=2, strides=2, padding='same', name='pooling_3')

            e_conv_7, e_batch_7, e_activ_7 = self._conv_bn_layer(pool_3, n_filters=self.num_filters, filter_scale=8,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='e_conv_bn', name_postfix='_7')
            e_conv_8, e_batch_8, e_activ_8 = self._conv_bn_layer(e_activ_7, n_filters=self.num_filters, filter_scale=8,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='e_conv_bn', name_postfix='_8')

            self.nets.extend([e_conv_1, e_conv_2, pool_1, e_conv_3, e_conv_4, pool_2, e_conv_5, e_conv_6, pool_3,
                              e_conv_7, e_conv_8])

        with tf.name_scope('decoder'):
            up_conv_1 = self.upconv_f(e_activ_8, filters=4 * self.num_filters, kernel_size=2, strides=2,
                                      name='upconv_1')
            concat_1 = tf.concat([p3, up_conv_1], axis=-1, name='concat_1')
            d_conv_1, d_batch_1, d_activ_1 = self._conv_bn_layer(concat_1, n_filters=self.num_filters, filter_scale=4,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='d_conv_bn', name_postfix='_9')
            d_conv_2, d_batch_2, d_activ_2 = self._conv_bn_layer(d_activ_1, n_filters=self.num_filters, filter_scale=4,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='d_conv_bn', name_postfix='_10')

            up_conv_2 = self.upconv_f(d_activ_2, filters=2 * self.num_filters, kernel_size=2, strides=2,
                                      name='upconv_2')
            concat_2 = tf.concat([p2, up_conv_2], axis=-1, name='concat_2')
            d_conv_3, d_batch_3, d_activ_3 = self._conv_bn_layer(concat_2, n_filters=self.num_filters, filter_scale=2,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='d_conv_bn', name_postfix='_11')
            d_conv_4, d_batch_4, d_activ_4 = self._conv_bn_layer(d_activ_3, n_filters=self.num_filters, filter_scale=2,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='d_conv_bn', name_postfix='_12')

            up_conv_3 = self.upconv_f(d_activ_4, filters=self.num_filters, kernel_size=2, strides=2, name='upconv_3')
            concat_3 = tf.concat([p1, up_conv_3], axis=-1, name='concat_3')
            d_conv_5, d_batch_5, d_activ_5 = self._conv_bn_layer(concat_3, n_filters=self.num_filters, filter_scale=1,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='d_conv_bn', name_postfix='_13')
            d_conv_6, d_batch_6, d_activ_6 = self._conv_bn_layer(d_activ_5, n_filters=self.num_filters, filter_scale=1,
                                                                 filter_size=3, is_training=self.is_training,
                                                                 nonlin_f=self.nonlin_f, padding='same',
                                                                 name='d_conv_bn', name_postfix='_14')

            self.nets.extend([up_conv_1, concat_1, d_conv_1, d_conv_2, up_conv_2, concat_2, d_conv_3, d_conv_4,
                              up_conv_3, concat_3, d_conv_5, d_conv_6])

        with tf.name_scope('output'):
            output = tf.layers.conv2d(d_activ_6, filters=self.num_classes, kernel_size=3, activation=None, padding='same',
                                      name='output')
            self.nets.append(output)

        return output



class UNet_pt(NetworkBase, nn.Module):
    def __init__(self, network_type, loss, accuracy, lr, framework,  training, trainable_layers=None, num_filters=16,
                 optimizer='adam', nonlin='elu', num_classes=2, dropout=0.25):
        NetworkBase.__init__(self, network_type=network_type, loss=loss, accuracy=accuracy, framework=framework, lr=lr, training=training,
                             trainable_layers=trainable_layers, num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                             num_classes=num_classes, dropout=dropout)
        nn.Module.__init__(self)

        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(64 + 128, 64)
        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)

        self.conv_last = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )