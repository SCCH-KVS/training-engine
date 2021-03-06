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
    config.ref_steps = 3
    config.ref_patience = 3
    config.batch_size = 32
    config.num_epochs = 3
    config.loss = 'softmax'
    config.optimizer = 'sgd'
    config.gradcam_record = True
    config.gradcam_layers = 6
    config.gradcam_layers_max = 6
    config.mi_record = False
    config.gpu_load = 0.8
    config.num_classes = 10
    config.class_labels = [str(i) for i in range(10)]
    config.num_filters = 16
    config.upconv = 'upconv'
    config.nonlin = 'relu'
    config.task_type = 'classification'
    config.accuracy = 'percent'
    config.augmentation = {'flip_hor': False,
                           'flip_vert': False}
    config.data_split = 0.7
    config.long_summary = True
    config.trainable_layers = 'all'
    config.normalize = True
    config.zero_center = True
    config.dropout = 0.4
    # config.chpnt2load = r'E:\SCCH_PROJECTS\DL_implementaitons\00_templates\training_engine_git\experiments\ckpnt_logs\MNIST\1565681155.1648371\1565681155.1648371_split_0'
    config.chpnt2load = r'experiments/ckpnt_logs/MNIST/1565680688.729741/1565680688.729741_split_0'
    config.multi_task = False
    config.cross_val = 1
    config.framework = 'pytorch'
    config.experiment_path = None


    return config