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
    config.lr = 0.001  # changable
    config.lr_decay = 0.00001  # changable
    config.ref_steps = 5  # changable
    config.ref_patience = 3  # changable
    config.batch_size = 104  # changable
    config.num_epochs = 18  # changable
    config.loss = 'softmax'  # changable
    config.optimizer = 'adam'   #changable
    config.mi_record = False
    config.gpu_load = 0.7
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
    config.hyperband = False
    config.halving_proportion = 3  # min 2, default 3
    config.max_amount_resources = 20   # no of max epochs (min 2, max 1000)
    # lr and lr decay can be choices or continoous - if there is a change check and change
    # _get_random_numbers(self) and _get_configspace(self) in Train Runner
    config.lr_range = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # [0.1, 0.00001]
    config.lr_decay_range = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # [0.1, 0.00001]
    config.ref_steps_range = [0, 6]  # 0, number of epochs /3
    config.ref_patience_range = [0, 6]   # 0, number of epochs /3
    config.batch_size_range = [config.num_classes, 512]  # num_classes, max capacity of memory
    config.loss_range = ['softmax']  # ['softmax', 'sigmoid', 'margin', 'mse', 'mse_loss']
    config.accuracy_range = ['percent']  # ['mse', 'percent']
    config.optimizer_range = ['adam']  # ['adam', 'momentum', 'gradient']

    config.bohb = False
    config.bandwidth_factor = 3
    config.top_n_percent = 0.5  # between 0.01 and 0.99
    config.num_samples = 64
    config.min_bandwidth = 0.01

    return config
