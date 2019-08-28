#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 2/20/2019 1:07 PM $ 
# by : shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------
import importlib
import tensorflow as tf
import torch
import tf2onnx


import onnx
from onnx_tf.backend import prepare

from utils.VisdomLogger import *
from utils.DataParser import DataParser
from network.wrappers import ConvNet, VGG19, UNet, LeNet, VGG16, ResNet


class NetRunner:
    """

    """
    def __init__(self, args=None, experiment_id=None):
        self.X = None
        self.y = None
        self.X_valid = None
        self.y_valid = None
        self.num_classes = None

        self._parse_config(args, experiment_id)

    def _parse_config(self, args, experiment_id):
        """
        Read parameters from config files
        :param args:
        :param experiment_id:
        :return:
        """
        if not args:
            if experiment_id:
                config = importlib.import_module('configs.config_' + experiment_id)
                args = config.load_config()
            else:
                raise ValueError('No arguments or configuration data given')
        # Mandatory parameters for all architectures
        self.network_type = args.net
        self.is_training = args.training_mode
        self.data_dir = args.data_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.trainlog_dir = args.trainlog_dir
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.ref_steps = args.ref_steps
        self.ref_patience = args.ref_patience
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.loss_type = args.loss
        self.accuracy_type = args.accuracy
        self.optimizer = args.optimizer
        self.dropout = args.dropout
        self.gradcam_record = args.gradcam_record
        self.gradcam_layers = args.gradcam_layers
        self.gradcam_layers_max = args.gradcam_layers_max
        self.gpu_load = args.gpu_load
        self.num_filters = args.num_filters
        self.upconv = args.upconv
        self.nonlin = args.nonlin
        self.task_type = args.task_type
        self.long_summary = args.long_summary
        self.multi_task = args.multi_task
        self.framework = args.framework
        self.cross_val = args.cross_val
        self.experiment_path = args.experiment_path
        self.chpnt2load = args.chpnt2load
        self.hyperband = args.hyperband

        if args.hyperband:
            self.halving_proportion = args.halving_proportion
            self.max_amount_resources = args.max_amount_resources
            self.lr_range = args.lr_range
            self.lr_decay_range = args.lr_decay_range
            self.ref_steps_range = args.ref_steps_range
            self.ref_patience_range = args.ref_patience_range
            self.batch_size_range = args.batch_size_range
            self.loss_range = args.loss_range
            self.accuracy_range = args.accuracy_range
            self.optimizer_range = args.optimizer_range
        if not self.is_training:
            self.class_labels = args.class_labels
        if args.data_set:
            self.data_set = args.data_set
        else:
            self.data_set = None
        if args.data_split:
            self.data_split = args.data_split
        else:
            self.data_split = None
        if args.data_file:
            self.data_file = args.data_file
        else:
            self.data_file = None
        if args.image_size:
            self.img_size = args.image_size
        else:
            self.img_size = None
        if args.num_classes:
            self.num_classes = args.num_classes
        else:
            self.num_classes = None
        if args.augmentation:
            self.augmentation_dict = args.augmentation
        else:
            self.augmentation_dict = None
        if args.trainable_layers:
            self.trainable_layers = args.trainable_layers
        else:
            self.trainable_layers = None
        if args.normalize:
            self.normalize = args.normalize
        else:
            self.normalize = None
        if args.zero_center:
            self.zero_center = args.zero_center
        else:
            self.zero_center = None

        self._initialize_data()

    def _initialize_data(self):
        data_parser = DataParser(self.data_set, self.data_file, self.data_dir, self.num_classes, self.img_size,
                                 self.normalize, self.zero_center, self.data_split, self.cross_val, self.task_type,
                                 self.is_training, self.multi_task, self.experiment_path, self.framework)
        if self.is_training:
            self.h5_data_file = data_parser.get_file_name()
            self.json_log = data_parser.get_log_name()
            self.data_size = data_parser.get_data_size()
            self.timestamp = data_parser.get_timestamp()
            self.ckpnt_path = data_parser.get_ckpnt_path()
            self.tr_path = data_parser.get_tr_path()
            self.hyperband_path = data_parser.get_hyperband_path()
        else:
            self.inference_X, self.inference_y = data_parser.get_inference_data()

    def build_tensorflow_pipeline(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if self.task_type is 'classification':
                in_shape = [None, self.img_size[0], self.img_size[1], self.img_size[2]]
                gt_shape = [None, self.num_classes]
                self._in_data = tf.placeholder(tf.float32, shape=in_shape, name='Input_train')
                self._gt_data = tf.placeholder(tf.float32, shape=gt_shape, name='GT_train')
            elif self.task_type is 'segmentation':
                in_shape = [None, self.img_size[0], self.img_size[1], self.img_size[2]]
                gt_shape = [None, self.img_size[0], self.img_size[1], self.num_classes]
                self._in_data = tf.placeholder(tf.float32, shape=in_shape, name='Input_train')
                self._gt_data = tf.placeholder(tf.float32, shape=gt_shape, name='GT_train')
            elif self.task_type is 'gan':
                in_shape = NotImplementedError
                gt_shape = NotImplementedError
            elif self.task_type is 'detection':
                in_shape = [None, self.img_size[0], self.img_size[1], self.img_size[2]]
                gt_shape = [None, 4+self.num_classes, None]
                self._in_data = tf.placeholder(tf.float32, shape=in_shape, name='Input_train')
                self._gt_data = tf.placeholder(tf.float32, shape=gt_shape, name='GT_train')
            else:
                raise ValueError('Task not supported')

            queue = tf.FIFOQueue(self.data_size * self.num_epochs, [tf.float32, tf.float32], name='queue')
            self.enqueue_op = queue.enqueue([self._in_data, self._gt_data])
            self.in_data, self.gt_data = queue.dequeue()
            self.in_data.set_shape(in_shape)
            self.gt_data.set_shape(gt_shape)

            self.learning_rate = tf.placeholder(tf.float32, name='Learning_rate')
            self.training_mode = tf.placeholder(tf.bool, name='Mode_train')

            self.epoch_loss = tf.placeholder(tf.float32, name='Epoch_loss')
            self.epoch_accuracy = tf.placeholder(tf.float32, name='Epoch_accuracy')
            self.loss_plot = tf.placeholder(tf.float32, name='Epoch_loss')
            self.learning_rate_plot = tf.placeholder(tf.float32, name='Epoch_learn')

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            self.network = self._pick_model()
            self.pred_output = self.network.build_net(self.in_data)
            self.global_step = tf.train.get_or_create_global_step()

            with tf.control_dependencies(self.update_ops):
                self.loss = self.network.return_loss(y_pred=self.pred_output, y_true=self.gt_data)
                self.train_op = self.network.return_optimizer(global_step=self.global_step)
                self.accuracy = self.network.return_accuracy(y_pred=self.pred_output, y_true=self.gt_data,
                                                             b_s=self.batch_size, net_type=self.task_type,
                                                             loss_type=self.loss_type)

            self.accuracy_plot = tf.placeholder(tf.float32, name='Epoch_accuracy')

            if self.gradcam_record:
                if self.gradcam_layers > len(self.network.nets) or self.gradcam_layers < 1:
                    self.gradcam_record = False
                    print('Wrong layer number, Grad-CAM disabled')
                else:
                    gradcam_size = None
                    self.gradcam_img = tf.placeholder(tf.uint8, shape=gradcam_size, name='GradCAM_merged_images')
                    self.gradcam_g_img = tf.placeholder(tf.uint8, shape=gradcam_size,
                                                        name='GuidedGradCAM_merged_images')
                    self.gb_img = tf.placeholder(tf.uint8, shape=gradcam_size, name='GuidedGradCAM_merged_images')
                    self.gb_grads = tf.gradients(self.loss, self.in_data)[0]
                    self.conv_outs = []
                    self.first_derivatives = []
                    self.second_derivatives = []
                    self.third_derivatives = []

                    for cl in range(self.gradcam_layers):
                        cli = len(self.network.nets) - cl - 1
                        self.conv_outs.append(self.network.nets[cli])
                        gs = tf.gradients(self.loss, self.network.nets[cli])[0]
                        self.first_derivatives.append(tf.exp(self.loss) * gs)
                        self.second_derivatives.append(tf.exp(self.loss) * gs * gs)
                        self.third_derivatives.append(tf.exp(self.loss) * gs * gs * gs)

            tf.add_to_collection('mode_train', self.training_mode)
            tf.add_to_collection('inputs_train', self.in_data)
            tf.add_to_collection('outputs_train', self.pred_output)
            tf.add_to_collection('learn_rate', self.learning_rate)
            # collections for Grad-CAM
            if self.gradcam_record:
                tf.add_to_collection('y_true', self.gt_data)
                tf.add_to_collection('gradcam_in_data', self.in_data)
                tf.add_to_collection('gradcam_gb_grads', self.gb_grads)
                for i in range(len(self.conv_outs)):
                    tf.add_to_collection('gradcam_conv_outs_{}'.format(i), self.conv_outs[i])
                    tf.add_to_collection('gradcam_first_derivatives_{}'.format(i), self.first_derivatives[i])
                    tf.add_to_collection('gradcam_second_derivatives_{}'.format(i), self.second_derivatives[i])
                    tf.add_to_collection('gradcam_third_derivatives_{}'.format(i), self.third_derivatives[i])
            self.net_params = [self.training_mode, self.train_op, self.loss, self.accuracy,
                               self.learning_rate, self.pred_output]
            # Extend net_params with Grad-CAM things
            if self.gradcam_record:
                # net_params + [6][[0]input image + [1]grads + [2]conv_layers + [3,4,5]derivatives]
                self.net_params.append(
                    [self.in_data, self.gb_grads, self.conv_outs, self.first_derivatives, self.second_derivatives,
                     self.third_derivatives])
            self.summ_params = [self.learning_rate_plot, self.loss_plot, self.accuracy_plot]

            self.graph_op = tf.global_variables_initializer()

    def build_pytorch_pipeline(self):
        cuda_en = torch.cuda.is_available()
        if self.gpu_load != 0 and cuda_en:
            self.device = torch.device('cuda')
            print("CUDA detecvted")
        else:
            self.device = torch.device('cpu')
            print("CPU detecvted")


        self.learning_rate = self.lr
        self.network = self._pick_model().to(self.device)

        self.train_op = self.network.return_optimizer(net_param=self.network.parameters())

        self.graph_op = VisdomPlotter(env_name='Training_Procedure')

    def _pick_model(self):
        """
        Pick a deep model specified by self.network_type string
        :return:
        """
        if self.framework == "tensorflow":
            if self.network_type.endswith('.onnx'):
                print('onnx!')
                return 'onxx'
            else:
                if self.network_type == 'ConvNet':
                    return ConvNet.ConvNet(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                           training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                              trainable_layers=self.trainable_layers)
                elif self.network_type == 'UNet':
                    return UNet.UNet(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate,  framework=self.framework, training=self.training_mode,
                                     trainable_layers=self.trainable_layers, num_classes=self.num_classes)
                elif self.network_type == 'VGG19':
                            return VGG19.VGG19_tf(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate,
                                               training=self.training_mode, framework=self.framework, num_classes=self.num_classes, trainable_layers=self.trainable_layers)
                elif self.network_type == 'LeNet':
                    return LeNet.LeNet(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                           training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                              trainable_layers=self.trainable_layers)
                elif self.network_type == 'VGG16':
                    return VGG16.VGG16(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate,
                                     framework=self.framework, training=self.training_mode,
                                     trainable_layers=self.trainable_layers, num_classes=self.num_classes)
                elif self.network_type == 'ResNet':
                    return ResNet.ResNet(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate,
                                     framework=self.framework, training=self.training_mode,
                                     trainable_layers=self.trainable_layers, num_classes=self.num_classes)
                else:
                    raise ValueError('Architecture does not exist')
        elif self.framework == "pytorch":
            if self.network_type == 'ConvNet':
                return ConvNet.ConvNet_pt(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                       training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                          trainable_layers=self.trainable_layers)
            elif self.network_type == 'VGG19':
                return VGG19.VGG19_pt(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                       training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                          trainable_layers=self.trainable_layers)
            elif self.network_type == 'UNet':
                return UNet.UNet_pt(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                       training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                          trainable_layers=self.trainable_layers)
            elif self.network_type == 'LeNet':
                return LeNet.LeNet_pt(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                       training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                          trainable_layers=self.trainable_layers)
            elif self.network_type == 'VGG16':
                return VGG16.VGG16_pt(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                       training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                          trainable_layers=self.trainable_layers)
            elif self.network_type == 'ResNet':
                return ResNet.ResNet_pt(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate, framework=self.framework,
                                       training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin, num_classes=self.num_classes,
                                          trainable_layers=self.trainable_layers)
            else:
                raise ValueError('Architecture does not exist')

    def _initialize_short_summary(self):
        """
        Tensorboard scope initialization
        :return:
        """
        loss_sum = tf.summary.scalar('Loss_function', self.loss_plot)
        lr_summ = tf.summary.scalar("Learning_rate", self.learning_rate_plot)
        # more then one accuracy specified
        if isinstance(self.accuracy_type, list) and (len(self.accuracy_type) >= 2):
            acc_summ = tf.stack([tf.summary.scalar('{:s}_Accuracy_label_{:d}'.format(self.accuracy_type[k], i),
                                                   self.accuracy_plot[k][i]) for k in range(len(self.accuracy_type))
                                 for
                                 i in range(self.num_classes)])
        # only one accuracy specified
        else:
            if isinstance(self.accuracy_type, list):
                self.accuracy_type = self.accuracy_type[0]
            if self.task_type == 'classification':
                acc_summ = tf.summary.scalar('{:s}_accuracy'.format(self.accuracy_type), self.accuracy_plot)
            elif self.task_type == 'segmentation':
                acc_summ = tf.summary.scalar('{:s}_accuracy'.format(self.accuracy_type), self.accuracy_plot)
                # acc_summ = tf.stack([tf.summary.scalar('{:s}_accuracy_label_{:d}'.format(self.accuracy_type, i),
                #                                        self.accuracy_plot[i]) for i in range(self.num_classes)])
        summary_op = tf.summary.merge([loss_sum, lr_summ, acc_summ])
        return summary_op

    def _initialize_long_summary(self):
        img_summ = tf.summary.image("Original_Image", self.in_data, self.batch_size)
        if self.task_type == 'segmentation':
            gt_summ = tf.summary.image("GT_Mask",
                                       self._convert_mask_data(self.batch_size, self.gt_data, self.num_classes),
                                       max_outputs=1)
            if self.loss_type == 'softmax':
                pred_summ = tf.summary.image("Predicted_Mask",
                                             self._convert_mask_data(self.batch_size,
                                                                     tf.nn.softmax(self.pred_output),
                                                                     self.num_classes), max_outputs=1)
            elif self.loss_type == 'sigmoid':
                pred_summ = tf.summary.image("Predicted_Mask",
                                             self._convert_mask_data(self.batch_size,
                                                                     tf.nn.sigmoid(self.pred_output),
                                                                     self.num_classes), max_outputs=1)
        elif self.task_type == 'classification':
            gt_summ = tf.summary.text('Input_labels',
                                      tf.as_string(
                                          tf.reshape(tf.argmax(self.gt_data, axis=1), [-1, self.batch_size])))
            pred_summ = tf.summary.text('Predicted_labels', tf.as_string(
                tf.reshape(tf.argmax(self.pred_output, axis=1), [-1, self.batch_size])))
        # Setup Grad-CAM summaries
        if self.gradcam_record:
            gradcam_summs = []
            g_gradcam_summs = []
            for cl in range(self.gradcam_layers):
                l_idx = self.gradcam_layers_max - cl
                gradcam_summs.append(
                    tf.summary.image("GradCAM-{}".format(l_idx), tf.reverse(self.gradcam_img[cl], [-1]),
                                     self.batch_size))
                g_gradcam_summs.append(
                    tf.summary.image("GuidedGradCAM-{}".format(l_idx), tf.reverse(self.gradcam_g_img[cl], [-1]),
                                     self.batch_size))
                if cl == 0:
                    gb_summ = tf.summary.image("GuidedBackpropagation", tf.reverse(self.gb_img, [-1]),
                                               self.batch_size)

        #         # add weights and biases to summary
        # <<<<<<< .mine
        #         # hist_summ = tf.stack(
        #         #     [tf.summary.histogram(tf.trainable_variables()[i].name[:-2] + '_train', tf.trainable_variables()[i]) for i
        #         #      in range(len(tf.trainable_variables()))])
        #         summary_op = tf.summary.merge([img_summ, gt_summ, pred_summ])
        #
        #         # summary_op = tf.summary.merge([img_summ, gt_summ, pred_summ, hist_summ])
        #         # summary_op = tf.summary.merge([img_summ, pred_summ, hist_summ])
        # ||||||| .r168
        #         hist_summ = tf.stack(
        #             [tf.summary.histogram(tf.trainable_variables()[i].name[:-2] + '_train', tf.trainable_variables()[i]) for i
        #              in range(len(tf.trainable_variables()))])
        #         summary_op = tf.summary.merge([img_summ, gt_summ, pred_summ, hist_summ])
        #         # summary_op = tf.summary.merge([img_summ, pred_summ, hist_summ])
        # =======
        hist_summ = tf.stack(
            [tf.summary.histogram(tf.trainable_variables()[i].name[:-2] + '_train', tf.trainable_variables()[i]) for
             i
             in range(len(tf.trainable_variables()))])
        m_summs = [img_summ, gt_summ, pred_summ, hist_summ]
        summary_op = tf.summary.merge(m_summs)
        # >>>>>>> .r281

        if self.gradcam_record:
            return summary_op, tf.summary.merge([gb_summ, gradcam_summs, g_gradcam_summs])
        else:
            return summary_op, None