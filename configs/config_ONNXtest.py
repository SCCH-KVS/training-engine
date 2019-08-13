#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 5/9/2019 3:49 PM $ 
# by : Shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------

#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 3/21/2019 8:30 AM $
# by : Shepeleva $
# SVN  $
#

# --- imports -----------------------------------------------------------------


from configs.config import ConfigFlags


def load_config():
    config = ConfigFlags().return_flags()

    config.net = r'E:\o156.onnx'

    config.autotune = False
    config.training_mode = True

    config.data_set = 'CIFAR10'
    config.image_size = [32, 32, 3]
    config.data_folder = None
    config.data_file = None


    config.lr = 0.001
    config.lr_decay = 0.1
    config.ref_steps = 10
    config.ref_patience = 10
    config.batch_size = 128
    config.num_epochs = 2
    config.loss = 'softmax'
    config.optimizer = 'sgd'
    config.gradcam_record = True
    config.gradcam_layers = 6
    config.gradcam_layers_max = 6
    config.mi_record = False
    config.gpu_load = 0.8
    config.num_classes = 10
    config.class_labels = [i for i in range(9)]
    config.num_filters = 64
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
    config.chpnt2load = ''
    config.multi_task = False
    config.cross_val = 1
    config.framework = 'tensorflow'
    config.experiment_path = None


    return config