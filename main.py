#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 2/20/2019 1:02 PM $ 
# by : shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------

import sys
import importlib

from network.TrainRunner import TrainRunner
from network.InferenceRunner import InferenceRunner

EXPERIMENT_ID = 'ConvNet'


def run(experiment_id):
    config = importlib.import_module('configs.config_' + experiment_id)
    args = config.load_config()
    if args.training_mode:
        training = TrainRunner(experiment_id=experiment_id)
        training.start_training()
    else:
        inference = InferenceRunner(experiment_id=experiment_id)
        inference.start_inference()
        raise ValueError('Inference not implemented yet')


if __name__ == '__main__':
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) != 3:
        print('You are doing something wrong! We will show you ToyNet')
        experiment = EXPERIMENT_ID
    else:
        experiment = sys.argv[2]
        print('Running {}'.format(experiment))
    run(experiment)
