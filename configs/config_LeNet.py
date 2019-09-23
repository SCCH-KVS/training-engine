#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 19/08/2019 16:29 $
# by : scchwagner $
# SVN : $
#

# --- imports -----------------------------------------------------------------

from configs.config import ConfigFlags


def load_config():
    config = ConfigFlags().return_flags()

    config.net = 'LeNet'
    config.training_mode = True
    config.data_set = 'MNIST'
    config.image_size = [32, 32, 1]
    config.lr = 0.02419  # changable
    config.lr_decay = 0.05931  # changable
    config.ref_steps = 2  # changable
    config.ref_patience = 1  # changable
    config.batch_size = 124  # changable
    config.num_epochs = 20  # changable
    config.loss = 'sigmoid'  # changable
    config.optimizer = 'gradient'   #changable
    config.mi_record = False
    config.gpu_load = 0.85
    config.num_classes = 10
    config.class_labels = [str(i) for i in range(10)]
    config.upconv = 'upconv'
    config.nonlin = 'relu'
    config.task_type = 'classification'
    config.accuracy = 'percent'  # changable
    config.augmentation = {'flip_hor': False,
                           'flip_vert': False}
    config.data_split = 0.7
    config.long_summary = False
    config.trainable_layers = 'all'
    config.normalize = True
    config.zero_center = True
    #config.chpnt2load = r'experiments/ckpnt_logs/MNIST/1568115668.7565737/1568115668.7565737_split_0'
    config.chpnt2load = ''
    config.multi_task = False
    config.cross_val = 1
    config.framework = 'tensorflow'
    config.experiment_path = None

    # For no optimisation:
    # enter range with desired value e.g. [0, 0] for continuous
    # or fixed Value for discrete e.g. ['percent']
    # Epochs are definded over max_amount_resources
    config.hyperband = True
    config.halving_proportion = 3  # min 2, default 3
    config.max_amount_resources = 27   # no of max epochs (min 2, max 1000)
    config.lr_range = [0.00001, 0.1]  # [0.1, 0.00001]
    config.lr_decay_range = [0.00001, 0.1]  # [0.1, 0.00001]
    config.ref_steps_range = [1, int(config.max_amount_resources/3)]  # 0, number of epochs /3
    config.ref_patience_range = [1, int(config.max_amount_resources/3)]   # 0, number of epochs /3
    config.batch_size_range = [config.num_classes, 512]  # num_classes, max capacity of memory
    config.loss_range = ['softmax', 'sigmoid', 'margin']  # ['softmax', 'sigmoid', 'margin', 'mse', 'mse_loss']
    config.accuracy_range = ['mse', 'percent']  # ['mse', 'percent']
    config.optimizer_range = ['adam', 'momentum', 'gradient']  # ['adam', 'momentum', 'gradient']

    config.bohb = True
    config.bandwidth_factor = 3
    config.top_n_percent = 0.5  # between 0.01 and 0.99
    config.num_samples = 64
    config.min_bandwidth = 0.01

    return config
