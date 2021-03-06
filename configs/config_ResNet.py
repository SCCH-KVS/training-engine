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

    config.net = 'ResNet'
    config.training_mode = True
    config.data_set = 'CIFAR100'
    config.image_size = [32, 32, 3]
    config.lr = 0.1  # changable
    config.lr_decay = 0.2  # changable
    config.ref_steps = 3  # changable
    config.ref_patience = [60, 120, 160]  # changable
    config.batch_size = 128  # changable
    config.num_epochs = 200  # changable
    config.loss = 'cross-entropy'  # changable
    config.optimizer = 'margin'  #changable
    config.mi_record = False
    config.gpu_load = 0.8
    config.num_classes = 100
    config.filters = 64
    config.class_labels = [str(i) for i in range(100)]
    config.upconv = 'upconv'
    config.nonlin = 'relu'
    config.task_type = 'classification'
    config.accuracy = 'percent'  # changable
    config.augmentation = {'flip_hor': False,
                           'flip_vert': False}
    config.data_split = 0.7
    config.long_summary = True
    config.trainable_layers = 'all'
    config.normalize = True
    config.zero_center = True
    #config.chpnt2load = r'experiments/ckpnt_logs/MNIST/1566288949.3847256/1566288949.3847256_split_0'
    config.chpnt2load = ''
    config.multi_task = False
    config.cross_val = 1
    config.framework = 'pytorch'
    config.experiment_path = None

    config.hyperband = False
    config.bohb = False

    return config
