#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 11/16/2017 11:07 $
# by : shepeleva $
# SVN : $
#

# --- imports -----------------------------------------------------------------

import argparse


class ConfigFlags:
    def __init__(self):
        """

        """
        parser = argparse.ArgumentParser(description='Tensorflow DL training pipeline')
        # mandatory variables
        parser.add_argument('--net', help='Network',
                            type=str, default='ConvNet')
        parser.add_argument('--mode', help='training / test',
                            type=str, default='training')
        parser.add_argument('--data_set', help='Name/ID of the used data set',
                            type=str, default='MNIST')
        parser.add_argument('--data_dir', help='Learning data directory',
                            type=str, default='')
        parser.add_argument('--data_file', help='Data file',
                            type=str)
        parser.add_argument('--checkpoint_dir', help='Checkpoint directory',
                            type=str, default='ckpnt_dir')
        parser.add_argument('--trainlog_dir', help='Train records directory',
                            type=str, default='tr_log')
        parser.add_argument('--lr', help='Learning rate',
                            type=float, default=0.001)
        parser.add_argument('--lr_decay', help='Learning rate decay rate',
                            type=float, default=0.0005)
        parser.add_argument('--ref_steps', help='Refinement Steps',
                            type=int, default=5)
        parser.add_argument('--ref_patience', help='Refinement Patience',
                            type=int, default=200)
        parser.add_argument('--batch_size', help='Batch size',
                            type=int, default=32)
        parser.add_argument('--num_epochs', help='Number of epochs',
                            type=int, default=100)
        parser.add_argument('--loss', help='Loss function',
                            type=str, default='mse')
        parser.add_argument('--optimizer', help='Optimizers: \n\t adam - adam optimizer '
                                                '\n\t gradient - gradient descent '
                                                '\n\t proximalgrad - proximal gradient descent ',
                            default='adam')
        parser.add_argument('--dropout', help='Dropout Rate to use',
                            type=float, default=0.25)
        parser.add_argument('--tb_record', help='Tensorboard records on/off',
                            type=bool, default=True)
        parser.add_argument('--darkon_record', help='Darkon tool',
                            type=bool, default=False)
        parser.add_argument('--gradcam_record', help='GradCam tool',
                            type=bool, default=False)
        parser.add_argument('--gradcam_layers', help='Number of inspected convolution layers',
                            type=int, default=1)
        parser.add_argument('--gradcam_layers_max', help='Number of convolution layers',
                            type=int, default=3)
        parser.add_argument('--mi_record', help='MI tool',
                            type=bool, default=False)
        parser.add_argument('--gpu_load', help='GPU load percentage [0.1 : 1]',
                            type=int, default=0.8)
        # optional variables
        parser.add_argument('--image_size', help='Image size',
                            type=list, default=[128, 128, 1])
        parser.add_argument('--num_classes', help='Number of labels',
                            type=int, default=2)
        parser.add_argument('--num_filters', help='Number of filters',
                            type=int)
        # todo actually implement following parameters in NetRunner
        parser.add_argument('--filter_size', help='Filter size',
                            type=int)
        parser.add_argument('--pool_size', help='Pool size',
                            type=int)
        parser.add_argument('--stride_size', help='stride size',
                            type=int)
        parser.add_argument('--nonlin', help='Nonlinearity',
                            type=str)
        parser.add_argument('--upconv', help='Up convolution type: upconv or upsampling',
                            type=str)
        parser.add_argument('--multi_task', help="multiple task, i.e. different loss functions",
                            type=bool)
        self.args = parser.parse_args()

    def return_flags(self):
        """

        :return:
        """
        return self.args
