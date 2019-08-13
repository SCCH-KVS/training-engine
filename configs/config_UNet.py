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

    config.autotune = False

    config.net = 'UNet'
    config.training_mode = True
    config.data_set = 'rubble_sample'
    config.data_dir = r'F:\Datasets\Rubblemaster_segmentation'
    config.image_size = [1024, 1024, 1]

    config.lr = 0.001
    config.lr_decay = 0.1
    config.ref_steps = 3
    config.ref_patience = 3
    config.batch_size = 2
    config.num_epochs = 100
    config.loss = 'dice_jaccard'
    config.optimizer = 'sgd'
    config.gradcam_record = True
    config.gradcam_layers = 6
    config.gradcam_layers_max = 6
    config.mi_record = False
    config.gpu_load = 0.8
    config.num_classes = 1
    config.class_labels = [i for i in range(1)]
    config.num_filters = 8
    config.upconv = 'upconv'
    config.nonlin = 'relu'
    config.task_type = 'segmentation'
    config.accuracy = 'IoU'
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
    config.framework = 'pytorch'
    config.experiment_path = None


    return config