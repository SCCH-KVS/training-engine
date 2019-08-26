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
import tf2onnx
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# import tensorflow.contrib.slim as slim

from network.NetRunner import NetRunner
from utils.BatchIterator import BatchIterator
from utils.utils import log_loss_accuracy
from utils.gradcam.GradCAM import grad_cam_plus_plus as gradcam


import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.onnx as torch_onnx



class TrainRunner(NetRunner):
    def __init__(self, args=None, experiment_id=None):
        super().__init__(args, experiment_id)

    def start_training(self):
        """
        Start Neural Network training
        :return:
        """
        # training initialisation
        # with tf.device('/cpu:0'):
        # self._initialize_data()
        validation_scores = self._initialize_training()
        return validation_scores

    def _initialize_training(self):
        if self.framework == 'tensorflow':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            self.build_tensorflow_pipeline()
            valid_loss_final = self._run_tensorflow_pipeline()
            return valid_loss_final
        elif self.framework == 'pytorch':
            self.build_pytorch_pipeline()
            valid_loss_final =self._run_pytorch_pipeline()
            return valid_loss_final
        else:
            raise ValueError('Framework is not supported')

    def _run_tensorflow_pipeline(self):
        tf.reset_default_graph()

        if self.gpu_load != 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_load)
            config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
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
                # if not os.path.isdir(os.path.join(self.tr_path)):
                #     os.mkdir(os.path.join(self.tr_path, self.data_set, self.timestamp))

                summary_log_file_train = os.path.join(self.tr_path,  'train', self.timestamp + '_split_' + str(split))
                train_summary_writer = tf.summary.FileWriter(summary_log_file_train, sess.graph)

                summary_log_file_valid = os.path.join(self.tr_path, 'valid', self.timestamp + '_split_' + str(split))
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
                    for i in tqdm(train_generator, total=train_lim, unit=' steps', desc='Epoch {:d} train'.format(epoch), disable=False):
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
                    for i in tqdm(valid_generator, total=valid_lim, unit=' steps', desc='Epoch {:d} valid'.format(epoch), disable=False):
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
                            # if not os.path.isdir(os.path.join(self.ckpnt_path, self.data_set, self.timestamp)):
                            #     os.mkdir(os.path.join(self.ckpnt_path, self.data_set, self.timestamp))
                            #
                            # model_file_path = os.path.join('experiments', 'ckpnt_logs', self.data_set, self.timestamp, self.timestamp + '_split_' + str(split))

                            model_file_path = os.path.join(self.ckpnt_path,
                                                           self.timestamp + '_split_' + str(split))
                            if not os.path.exists(model_file_path):
                                os.makedirs(model_file_path)

                            saver.save(sess, "{}/model.ckpt".format(model_file_path), global_step=epoch)
                            # tf.train.write_graph(sess.graph.as_graph_def(), model_file_path, 'tensorflowModel.pbtxt',
                            #                      as_text=True)
                            #
                            # onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=["Input_train:0"],
                            #                                              output_names=["output:0"])
                            # model_proto = onnx_graph.make_model("test")
                            # with open("{}/model.onnx".format(model_file_path), "wb") as f:
                            #     f.write(model_proto.SerializeToString())

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
                                    model_file_path = os.path.join(self.ckpnt_path,
                                                                   self.timestamp + '_split_' + str(split))
                                    if not os.path.exists(model_file_path):
                                        os.makedirs(model_file_path)

                                    saver.save(sess, "{}/model.ckpt".format(model_file_path), global_step=epoch)

                                    # onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                    #                                              input_names=["Input_train:0"],
                                    #                                              output_names=["output:0"])
                                    # model_proto = onnx_graph.make_model("test")
                                    # with open("{}/model.onnx".format(model_file_path), "wb") as f:
                                    #     f.write(model_proto.SerializeToString())
                                    # sess.close()
                                    return prev_loss
                                    # break
                    gc.collect()
            coord.request_stop()
            coord.join(enqueue_threads)

        h5_file.close()


    def _run_pytorch_pipeline(self):
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

        for split in range(len(cross_val_split) - 1):
            print("Cross validation split {} of {}".format(str(split + 1), str(len(cross_val_split) - 1)))
            event_time = time.time()
            global_step = 0

            # TODO: add training and validation log summary

            learn_rate = self.lr
            prev_loss = np.inf
            ref_counter = 0
            ref_iter_count = 0
            total_recall_counter_train = 0
            total_recall_counter_valid = 0

            # liveloss = PlotLosses()

            # loss_plot_path = os.path.join(self.checkpoint_dir, 'prj_'+str(self.project_id), 'training_loss')
            # loss_plot_path = os.path.join(self.experiment_path, 'training_loss')
            # if not os.path.exists(loss_plot_path):
            #     os.makedirs(loss_plot_path)


            rp = ReduceLROnPlateau(self.train_op, factor=self.lr_decay, patience=self.ref_patience)

            for epoch in range(1, self.num_epochs + 1):
                train_generator = BatchIterator(h5_file, train_split[split], self.img_size, self.num_classes, self.batch_size,
                                                        self.task_type, self.is_training)

                train_lim = train_generator.get_max_lim()
                valid_generator = BatchIterator(h5_file, valid_split[split], self.img_size, self.num_classes, self.batch_size,
                                                        self.task_type, self.is_training)
                valid_lim = valid_generator.get_max_lim()
                epoch_loss_train = 0
                epoch_loss_valid = 0

                epoch_accur_train = 0
                epoch_accur_valid = 0

                epoch_duration_train = 0
                epoch_duration_valid = 0

                # print("image dimentions")
                # print([self.img_size[0], self.img_size[1], self.img_size[2]])

                for i in tqdm(train_generator, total=train_lim, unit=' steps', desc='Epoch {:d} train'.format(epoch)):
                    total_recall_counter_train += 1
                    # start_time = time.time()
                    ff = list(i)

                    outputs = self.network(torch.from_numpy(np.array([np.reshape(f[0], [self.img_size[0], self.img_size[1], self.img_size[2]]) for f
                                            in ff]).transpose((0,3,1,2))).float().to(self.device))
                    loss = self.network.return_loss(outputs, torch.from_numpy(np.array([f[1] for f in ff])).long().to(self.device)).to(self.device)
                    self.train_op.zero_grad()
                    loss.backward()
                    self.train_op.step()

                    train_step_accuracy = int(self.network.return_accuracy(outputs, torch.from_numpy(np.array([f[1] for f in ff])).to(self.device),  self.batch_size,
                                                self.task_type).to(self.device))

                    epoch_loss_train += loss.item()
                    epoch_accur_train += train_step_accuracy/self.batch_size

                for i in tqdm(valid_generator, total=valid_lim, unit=' steps', desc='Epoch {:d} valid'.format(epoch)):
                    total_recall_counter_valid += 1
                    # start_time = time.time()
                    ff = list(i)

                    outputs = self.network(torch.from_numpy(np.array([np.reshape(f[0], [self.img_size[0], self.img_size[1], self.img_size[2]]) for f
                                            in ff]).transpose((0,3,1,2))).float().to(self.device))
                    loss = self.network.return_loss(outputs, torch.from_numpy(np.array([f[1] for f in ff])).long().to(self.device)).to(self.device)

                    valid_step_accuracy = int(self.network.return_accuracy(outputs, torch.from_numpy(np.array([f[1] for f in ff])).to(self.device), 1,
                                                self.task_type).to(self.device))

                    epoch_loss_valid += loss.item()
                    epoch_accur_valid += valid_step_accuracy/self.batch_size

                train_aver_loss = epoch_loss_train / train_lim
                valid_aver_loss = epoch_loss_valid / valid_lim

                print(epoch_accur_train)

                if self.task_type == 'classification':
                    epoch_accur_train = epoch_accur_train / train_lim
                    epoch_accur_valid = epoch_accur_valid / valid_lim
                    # epoch_acc_str_tr = log_loss_accuracy(epoch_accur_train, self.accuracy_type, self.task_type,
                    #                                      self.num_classes)
                    # epoch_acc_str_val = log_loss_accuracy(epoch_accur_valid, self.accuracy_type, self.task_type,
                    #                                       self.num_classes)
                elif self.task_type == 'segmentation':
                    epoch_accur_train = epoch_accur_train / train_lim
                    epoch_accur_valid = epoch_accur_valid / valid_lim
                    # epoch_acc_str_tr = log_loss_accuracy(epoch_accur_train, self.accuracy_type, self.task_type,
                    #                                      self.num_classes)
                    # epoch_acc_str_val = log_loss_accuracy(epoch_accur_valid, self.accuracy_type, self.task_type,
                    #                                       self.num_classes)

                # print(
                #     '\nRESULTS: epoch {:d} train loss = {:.3f}, train accuracy : {:.3f} ({:.2f} sec) || '
                #     'valid loss = {:.3f}, valid accuracy : {:.3f} ({:.2f} sec)'
                #     .format(epoch, train_aver_loss, epoch_accur_train, epoch_duration_train, valid_aver_loss,
                #             epoch_accur_valid, epoch_duration_valid))

                prev_lr = self.train_op.param_groups[0]['lr']
                rp.step(valid_aver_loss)
                curr_lr = self.train_op.param_groups[0]['lr']
                # print("Prev lr: {} Curr lr: {}".format(prev_lr, curr_lr))
                #
                # print('update loss')
                # liveloss.update({
                #         'log loss': train_aver_loss,
                #         'val_log loss': valid_aver_loss,
                #         'accuracy': epoch_accur_train,
                #         'val_accuracy': epoch_accur_valid})
                # # print('draw loss')
                # figure = liveloss.draw()
                # print(figure)
                # print("saving figure")
                # # print(figure)

                self.graph_op.plot('loss', 'train', 'Class Loss', epoch, train_aver_loss)
                self.graph_op.plot('loss', 'valid', 'Class Loss', epoch, valid_aver_loss)

                self.graph_op.plot('acc', 'train', 'Class Accuracy', epoch, epoch_accur_train)
                self.graph_op.plot('acc', 'valid', 'Class Accuracy', epoch, epoch_accur_valid)

                # figure_file = os.path.join(loss_plot_path,'train_acur_plot.jpg')
                # figure.savefig(figure_file)

                print(
                    '\nRESULTS: epoch {:d} lr {:6f} train loss = {:.3f}, train accuracy : {:.3f} ({:.2f} sec) || '
                    'valid loss = {:.3f}, valid accuracy : {:.3f} ({:.2f} sec)'
                    .format(epoch, curr_lr, train_aver_loss, epoch_accur_train, epoch_duration_train, valid_aver_loss,
                            epoch_accur_valid, epoch_duration_valid))

                if prev_loss > valid_aver_loss:
                    if not os.path.isdir(self.checkpoint_dir):
                        os.mkdir(self.checkpoint_dir)

                    if not os.path.isdir(self.tr_path):
                        os.mkdir(self.tr_path)

                    model_file_path = os.path.join(self.ckpnt_path,
                                                   self.timestamp + '_split_' + str(split))
                    if not os.path.exists(model_file_path):
                        os.makedirs(model_file_path)

                    torch.save(self.network, "{}/model.pth".format(model_file_path))


                    # model_file_path = create_ckpt_data_pytorch(self.ckpnt_path, self.network_type, event_time)
                    # torch.save(self.network, model_file_path)
                    print('[SESSION SAVE] Epoch {:d}, loss: {:2f}, accur: {:2f}, lr: {:4f}'.format(epoch, valid_aver_loss, epoch_accur_valid, curr_lr))
                    prev_loss = valid_aver_loss
                    # ref_counter = 0
                if curr_lr < prev_lr:
                    ref_iter_count += 1
                    if ref_iter_count == self.ref_steps+1:
                        print('\nEarly stopping\n Saving last best model:')
                        final_network = torch.load(os.path.join(model_file_path, 'model.pth'))
                        # pprint.pprint(final_network.state_dict())
                        print("Importing model to ONNX")
                        dummy_input = Variable(torch.randn(1, *[self.img_size[2], self.img_size[0], self.img_size[1]]).to(self.device))
                        torch_onnx.export(final_network, dummy_input, os.path.join(model_file_path, 'model.onnx'), verbose=False)

                        print('Done')

                        return prev_loss
                gc.collect()
            print("Finalizing training")
            final_network = torch.load(os.path.join(model_file_path, 'model.pth'))
            # pprint.pprint(final_network.state_dict())
            print("Importing model to ONNX")
            # model_file_path_onnx = create_ckpt_data_onnx(self.ckpnt_path, self.network_type, event_time)
            dummy_input = Variable(torch.randn(1, *[self.img_size[2], self.img_size[0], self.img_size[1]]).to(self.device))
            torch_onnx.export(final_network, dummy_input, os.path.join(model_file_path, 'model.onnx'), verbose=False)

            print('Done')
            h5_file.close()
            return prev_loss