#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 2/20/2019 4:01 PM $ 
# by : shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------

import gc
import os
import time
import h5py
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# import tensorflow.contrib.slim as slim

from network.NetRunner import NetRunner
from utils.BatchIterator import BatchIterator
from utils.utils import log_loss_accuracy
from utils.gradcam.GradCAM import grad_cam_plus_plus as gradcam


class TrainRunner(NetRunner):
    def __init__(self, args=None, experiment_id=None):
        super().__init__(args, experiment_id)

    def start_training(self):
        """
        Start Neural Network training
        :return:
        """
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # training initialisation
        # with tf.device('/cpu:0'):
        # self._initialize_data()
        self._initialize_training()

    def _initialize_training(self):
        if self.framework is 'tensorflow':
            self.build_tensorflow_pipeline()
            self._run_tensorflow_pipeline()
        elif self.framework is 'pytorch':
            self.build_pytorch_pipeline()
        else:
            raise ValueError('Framework is not supported')

    def _run_tensorflow_pipeline(self):
        tf.reset_default_graph()

        if self.gpu_load != 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_load)
            # config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True,
            #                         device_count={"CPU": 4},
            #                         inter_op_parallelism_threads=4,
            #                         intra_op_parallelism_threads=5,)
            config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
            config.gpu_options.allow_growth = True
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            config = tf.ConfigProto(device_count={'CPU': 1})

        h5_file = h5py.File(self.h5_data_file, 'r')

        with open(self.json_log, 'r') as f:
            json_log = json.load(f)

        train_split = []
        valid_split = []
        cross_val_split = json_log['data_log']['cross_val_split']
        if len(cross_val_split) == 2:
            train_split.append(cross_val_split[0])
            valid_split.append(cross_val_split[1])
        else:
            for i in range(len(cross_val_split)):
                valid_split.append(cross_val_split[i])
                train_split.append([].extend(cross_val_split[j]) for j in range(len(cross_val_split)) if j != i)

        for split in range(len(cross_val_split)-1):
            print("Cross validation split % of %".format(str(split+1), str(len(cross_val_split)-1)))

            with tf.Session(graph=self.graph, config=config) as sess:
                event_time = time.time()
                global_step = tf.train.get_global_step(sess.graph)
                self.net_params.append(global_step)
                if self.long_summary:
                    long_summ, self.gradcam_summ = self._initialize_long_summary()
                    self.net_params.append(long_summ)
                short_summ = self._initialize_short_summary()
                self.summ_params.append(short_summ)
                if not os.path.isdir(os.path.join(self.tr_path, self.data_set, self.timestamp)):
                    os.mkdir(os.path.join(self.tr_path, self.data_set, self.timestamp))

                summary_log_file_train = os.path.join(self.tr_path, self.data_set, self.timestamp, 'train', self.timestamp + '_split_' + str(split))
                train_summary_writer = tf.summary.FileWriter(summary_log_file_train, sess.graph)

                summary_log_file_valid = os.path.join(self.tr_path, self.data_set, self.timestamp, 'valid', self.timestamp + '_split_' + str(split))
                valid_summary_writer = tf.summary.FileWriter(summary_log_file_valid, sess.graph)

                saver = tf.train.Saver(save_relative_paths=True)
                with tf.device('/cpu:0'):
                    self.graph_op.run()
                    learn_rate = self.lr
                    prev_loss = np.inf
                    ref_counter = 0
                    ref_iter_count = 0
                    total_recall_counter_train = 0
                    total_recall_counter_valid = 0
                    coord = tf.train.Coordinator()
                    enqueue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                for epoch in range(1, self.num_epochs + 1):
                    # print('\nEPOCH {:d}'.format(epoch))
                    with tf.device('/cpu:0'):
                        train_generator = BatchIterator(h5_file, train_split[split], self.img_size, self.num_classes, self.batch_size,
                                                        self.task_type, self.is_training)
                        train_lim = train_generator.get_max_lim()
                        # <<<<<<< .mine
                        #                     valid_generator = BatchIterator(h5_file, 'valid', self.img_size, self.num_classes, self.batch_size,
                        # ||||||| .r180
                        #                     valid_generator = BatchIterator(h5_file, 'valid', self.img_size, self.num_classes, 1,
                        # =======
                        #                     # TODO calidation self.batch_size -> 1 fix is required with long_summary
                        valid_generator = BatchIterator(h5_file, valid_split[split], self.img_size, self.num_classes, self.batch_size,
                                                        self.task_type, self.is_training)
                        valid_lim = valid_generator.get_max_lim()
                        epoch_loss_train = 0
                        epoch_loss_valid = 0
                        if self.task_type == 'classification':
                            epoch_accur_train = 0
                            epoch_accur_valid = 0
                        else:
                            epoch_accur_train = 0
                            epoch_accur_valid = 0
                        epoch_duration_train = 0
                        epoch_duration_valid = 0

                    # step_cout_train = 1
                    for i in tqdm(train_generator, total=train_lim, unit=' steps', desc='Epoch {:d} train'.format(epoch)):
                        with tf.device('/cpu:0'):
                            total_recall_counter_train += 1
                            start_time = time.time()
                            ff = list(i)
                            sess.run(self.enqueue_op, feed_dict={
                                self._in_data: [np.reshape(f[0], [self.img_size[0], self.img_size[1], self.img_size[2]]) for f
                                                in ff],
                                self._gt_data: [f[1] for f in ff]})
                            return_session_params = sess.run(self.net_params,
                                                             feed_dict={self.training_mode: True,
                                                                        self.learning_rate: learn_rate})
                        # Calculate Grad-CAM parameters
                        if self.gradcam_record and total_recall_counter_train % train_generator.get_max_lim() == 1:
                            gradcam_params = return_session_params[6]
                            merged_heatmaps = []
                            merged_gbs = []
                            for cl in range(self.gradcam_layers):
                                merged_heatmap, heatmap, merged_gb, gb = gradcam(gradcam_params, cl)
                                merged_heatmaps.append(merged_heatmap)
                                merged_gbs.append(merged_gb)
                            self.return_gradcam_summs = sess.run(self.gradcam_summ,
                                                                 feed_dict={self.gradcam_img: merged_heatmaps,
                                                                            self.gradcam_g_img: merged_gbs,
                                                                            self.gb_img: gb})
                        with tf.device('/cpu:0'):
                            if self.long_summary and total_recall_counter_train % train_generator.get_max_lim() == 1:
                                train_summary_writer.add_summary(return_session_params[-1], epoch)
                                # add Grad-CAM to the summary
                                if self.gradcam_record:
                                    train_summary_writer.add_summary(self.return_gradcam_summs, epoch)
                            train_step_loss = return_session_params[2]
                            train_step_accuracy = return_session_params[3]
                            train_step_duration = time.time() - start_time
                            epoch_loss_train += train_step_loss
                        # print(return_session_params[-1])
                        if self.task_type == 'classification':
                            epoch_accur_train += train_step_accuracy
                        else:
                            epoch_accur_train += train_step_accuracy
                        epoch_duration_train += train_step_duration
                        # print('Training current step: {:d}'.format(step_cout_train))
                        # step_cout_train += 1

                    # step_cout_valid = 1
                    for i in tqdm(valid_generator, total=valid_lim, unit=' steps', desc='Epoch {:d} valid'.format(epoch)):
                        with tf.device('/cpu:0'):
                            total_recall_counter_valid += 1

                            start_time = time.time()
                            ff = list(i)

                            sess.run(self.enqueue_op, feed_dict={
                                self._in_data: [np.reshape(f[0], [self.img_size[0], self.img_size[1], self.img_size[2]]) for f
                                                in ff],
                                self._gt_data: [f[1] for f in ff]})
                            return_session_params_valid = sess.run(self.net_params,
                                                                   feed_dict={self.training_mode: False,
                                                                              self.learning_rate: learn_rate})
                        # Calculate Grad-CAM parameters
                        if self.gradcam_record and total_recall_counter_valid % valid_generator.get_max_lim() == 1:
                            gradcam_params = return_session_params_valid[6]
                            merged_heatmaps = []
                            merged_gbs = []
                            for cl in range(self.gradcam_layers):
                                merged_heatmap, heatmap, merged_gb, gb = gradcam(gradcam_params, cl)
                                merged_heatmaps.append(merged_heatmap)
                                merged_gbs.append(merged_gb)
                            return_gradcam_summs = sess.run(self.gradcam_summ, feed_dict={self.gradcam_img: merged_heatmaps,
                                                                                          self.gradcam_g_img: merged_gbs,
                                                                                          self.gb_img: gb})
                        with tf.device('/cpu:0'):
                            if self.long_summary and total_recall_counter_valid % valid_generator.get_max_lim() == 1:
                                valid_summary_writer.add_summary(return_session_params_valid[-1], epoch)
                                # add Grad-CAM to the summary
                                if self.gradcam_record:
                                    valid_summary_writer.add_summary(return_gradcam_summs, epoch)

                        valid_step_loss = return_session_params_valid[2]
                        valid_step_accuracy = return_session_params_valid[3]
                        valid_step_duration = time.time() - start_time
                        epoch_loss_valid += valid_step_loss
                        if self.task_type == 'classification':
                            epoch_accur_valid += valid_step_accuracy
                        else:
                            epoch_accur_valid += valid_step_accuracy
                        epoch_duration_valid += valid_step_duration
                        # print('Validation current step: {:d}'.format(step_cout_valid))
                        # step_cout_valid += 1

                    # with tf.device('/cpu:0'):
                    train_aver_loss = epoch_loss_train / train_lim
                    valid_aver_loss = epoch_loss_valid / valid_lim

                    if self.task_type == 'classification':

                        epoch_accur_train = epoch_accur_train / train_lim
                        epoch_accur_valid = epoch_accur_valid / valid_lim

                        epoch_acc_str_tr = log_loss_accuracy(epoch_accur_train, self.accuracy_type, self.task_type,
                                                             self.num_classes, self.multi_task)
                        epoch_acc_str_val = log_loss_accuracy(epoch_accur_valid, self.accuracy_type, self.task_type,
                                                              self.num_classes, self.multi_task)
                    elif self.task_type == 'segmentation':
                        epoch_accur_train = epoch_accur_train / train_lim
                        epoch_accur_valid = epoch_accur_valid / valid_lim
                        epoch_acc_str_tr = log_loss_accuracy(epoch_accur_train, self.accuracy_type, self.task_type,
                                                             self.num_classes, self.multi_task)
                        epoch_acc_str_val = log_loss_accuracy(epoch_accur_valid, self.accuracy_type, self.task_type,
                                                              self.num_classes, self.multi_task)
                        # epoch_acc_str = 'Not implemented yet'

                    ret_train_epoch = sess.run(self.summ_params, feed_dict={self.learning_rate_plot: learn_rate,
                                                                            self.loss_plot: train_aver_loss,
                                                                            self.accuracy_plot: epoch_accur_train})
                    ret_valid_epoch = sess.run(self.summ_params, feed_dict={self.learning_rate_plot: learn_rate,
                                                                                self.loss_plot: valid_aver_loss,
                                                                                self.accuracy_plot: epoch_accur_valid})
                    train_summary_writer.add_summary(ret_train_epoch[-1], epoch)
                    valid_summary_writer.add_summary(ret_valid_epoch[-1], epoch)

                    with tf.device('/cpu:0'):
                        print(
                            '\nRESULTS: epoch {:d} train loss = {:.3f}, train accuracy : {:s} ({:.2f} sec) || valid loss = {:.3f}, valid accuracy : {:s} ({:.2f} sec)'
                            .format(epoch, train_aver_loss, epoch_acc_str_tr, epoch_duration_train, valid_aver_loss,
                                    epoch_acc_str_val, epoch_duration_valid))
                        if prev_loss > valid_aver_loss:
                            if not os.path.isdir(os.path.join(self.ckpnt_path, self.data_set, self.timestamp)):
                                os.mkdir(os.path.join(self.ckpnt_path, self.data_set, self.timestamp))

                            model_file_path = os.path.join('experiments', 'ckpnt_logs', self.data_set, self.timestamp, self.timestamp + '_split_' + str(split))
                            saver.save(sess, "{}/model.ckpt".format(model_file_path), global_step=epoch)

                            # onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, output_names=["output:0"])
                            # model_proto = onnx_graph.make_model("test")
                            # with open("{}/model.onnx".format(model_file_path), "wb") as f:
                            #     f.write(model_proto.SerializeToString())

                            print('[SESSION SAVE] Epoch {:d}, loss: {:2f}'.format(epoch, valid_aver_loss))
                            prev_loss = valid_aver_loss
                            ref_counter = 0
                        else:
                            ref_counter += 1
                            if ref_counter == self.ref_patience:
                                learn_rate *= self.lr_decay
                                print('\nDecreasing learning rate', learn_rate)
                                latest_ckpt = tf.train.latest_checkpoint(model_file_path)
                                saver.restore(sess, latest_ckpt)
                                print('\n[SESSION RESTORED]')
                                ref_counter = 0
                                ref_iter_count += 1
                                if ref_iter_count == self.ref_steps:
                                    print('\nEarly stopping')
                                    # sess.close()
                                    break
                    gc.collect()
            coord.request_stop()
            coord.join(enqueue_threads)

        h5_file.close()