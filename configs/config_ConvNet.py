#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 01/08/2018 16:29 $
# by : shepeleva $
# SVN : $
#

# --- imports -----------------------------------------------------------------

from configs.config import ConfigFlags


def load_config():
    config = ConfigFlags().return_flags()

    config.net = 'ConvNet'
    config.training_mode = False
    config.data_set = 'MNIST'
    config.image_size = [28, 28, 1]

    config.lr = 0.001
    config.lr_decay = 0.1
    config.ref_steps = 10
    config.ref_patience = 1
    config.batch_size = 32
    config.num_epochs = 1000
    config.loss = 'mse'
    config.optimizer = 'adam'
    config.gradcam_record = True
    config.gradcam_layers = 6
    config.gradcam_layers_max = 6
    config.mi_record = False
    config.gpu_load = 0.8
    config.num_classes = 10
    config.class_labels = [i for i in range(9)]
    config.num_filters = 16
    config.upconv = 'upconv'
    config.nonlin = 'relu'
    config.task_type = 'classification'
    config.accuracy = 'mse'
    config.augmentation = {'flip_hor': False,
                           'flip_vert': False}
    config.data_split = 0.7
    config.long_summary = True
    config.trainable_layers = 'all'
    config.normalize = True
    config.zero_center = True
    config.dropout = 0.4
    config.chpnt2load = ''
    config.multi_task = False
    config.cross_val = 1
    config.framework = 'tensorflow'
    config.experiment_path = None


    return config