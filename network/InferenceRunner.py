#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 2/25/2019 3:41 PM $ 
# by : shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------

import os
import torch
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import itertools
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from network.NetRunner import NetRunner
# from utils.BatchIterator import BatchIterator
# from utils.gradcam.GradCAM import grad_cam_plus_plus as gradcam
# from utils.gradcam.GradCAM import save_gradcam_data as save_gradcam

class InferenceRunner(NetRunner):
    def __init__(self, args=None, experiment_id=None):
        super().__init__(args, experiment_id)

    def start_inference(self):
        self._initialize_inference()

    def _initialize_inference(self):
        if self.framework is 'tensorflow':
            self._run_tensorflow_pipeline()
            self._result_eval()
        elif self.framework is 'pytorch':
            self._run_pytorch_pipeline()
            self._result_eval()
        else:
            raise ValueError('Framework is not supported')


    def _run_tensorflow_pipeline(self):
        tf.reset_default_graph()
        if self.chpnt2load:
            try:
                latest_ckpt = tf.train.latest_checkpoint(self.chpnt2load)
            except:
                raise ValueError('Mo such checkpoint is found ' + self.chpnt2load)
        else:
            raise ValueError("No checkpoint provided")

        saver = tf.train.import_meta_graph(latest_ckpt + '.meta')
        graph = tf.get_default_graph()
        if self.gpu_load != 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_load)
            config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
            config.gpu_options.allow_growth = True
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            config = tf.ConfigProto(device_count={'CPU': 1})

        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, latest_ckpt)

            X = tf.get_collection("inputs_train")[0]
            pred = tf.get_collection("outputs_train")[0]
            mode = tf.get_collection_ref("mode_train")[0]

            self.pred_result = []

            count = 0
            for i in tqdm(range(0, len(self.inference_X)), total=len(self.inference_X), unit=' steps',
                          desc='Inference'):
                if self.task_type == 'classification':
                    pred_result_ = sess.run(pred, feed_dict={
                        X: np.reshape(self.inference_X[i],
                                      [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False})
                    self.pred_result.extend(pred_result_)

                elif self.task_type == 'segmentation':
                    mask_pred = sess.run(pred, feed_dict={
                        X: np.reshape(self.inference_X[i],
                                      [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False})

                    mask = tf.nn.sigmoid(mask_pred).eval()
                    m_r = np.argmax(mask, axis=3)
                    f, (ax1, ax2) = plt.subplots(1, 2)

                    ax2.imshow(np.reshape(m_r, [self.img_size[0], self.img_size[1]]))
                    ax2.set(xlabel='generated segmentation')
                    ax1.imshow(np.reshape(self.inference_X[i], [self.img_size[0], self.img_size[1]]))
                    ax1.set(xlabel='original image')
                    f.savefig(os.path.join(self.data_dir, 'results', str(count) + '_.png'), figsize=(24, 18),
                              dpi=600)  # save the figure to file
                    plt.close(f)

                    count = count + 1
                else:
                    raise ValueError('No inference for this task type implemented')


    def _run_pytorch_pipeline(self):
        cuda_en = torch.cuda.is_available()
        if self.gpu_load != 0 and cuda_en:
            device = torch.device('cuda')
            print("CUDA detecvted")
        else:
            device = torch.device('cpu')
            print("CPU detecvted")


        model = torch.load(os.path.join(self.chpnt2load, 'model.pth'))
        model.to(device)
        model.eval()
        # return NotImplementedError
        self.pred_result = []

        count = 0
        for i in tqdm(range(0, len(self.inference_X)), total=len(self.inference_X), unit=' steps',
                      desc='Inference'):
            if self.task_type == 'classification':
                pred_result_ = model(torch.from_numpy(np.reshape(self.inference_X[i],
                                  [1, self.img_size[2], self.img_size[0], self.img_size[1]])).to(device,  dtype=torch.float)).to(device)
                self.pred_result.extend(pred_result_.cpu().detach().numpy())

            elif self.task_type == 'segmentation':
                mask_pred = model(torch.from_numpy(np.reshape(self.inference_X[i],
                                  [1, self.img_size[2], self.img_size[0], self.img_size[1]])).to(device,  dtype=torch.float)).to(device)

                # mask = tf.nn.sigmoid(mask_pred).eval()
                # m_r = np.argmax(mask, axis=3)
                # f, (ax1, ax2) = plt.subplots(1, 2)
                #
                # ax2.imshow(np.reshape(m_r, [self.img_size[0], self.img_size[1]]))
                # ax2.set(xlabel='generated segmentation')
                # ax1.imshow(np.reshape(self.inference_X[i], [self.img_size[0], self.img_size[1]]))
                # ax1.set(xlabel='original image')
                # f.savefig(os.path.join(self.data_dir, 'results', str(count) + '_.png'), figsize=(24, 18),
                #           dpi=600)  # save the figure to file
                # plt.close(f)
                #
                # count = count + 1
            else:
                raise ValueError('No inference for this task type implemented')


    def _result_eval(self):
        if self.task_type == 'classification':

            if self.loss_type == 'mse':
                m = 0
                y_pred = np.array(self.pred_result)
                y_true = np.array(self.inference_y)
                for i in range(0, len(y_pred)):
                    mse_test = np.sqrt(np.sum(((y_true[i] - y_pred[i]) ** 2) / 2))
                    # mse_test = sess.run(mse_loss(y_true[i], y_pred[i]))
                    m += mse_test
                    print("Prediction: {} GT: {} Score {}%".format(str(y_pred[i]), str(y_true[i]),
                                                                   int(mse_test * 100)))

                # mse_test = np.mean(np.sqrt(np.sum(((y_true - y_pred) ** 2)/2)))
                print("Overall: {}%".format(int(m / len(y_pred) * 100)))
            else:
                y_pred = np.argmax(np.array(self.pred_result).squeeze(), axis=1)
                y_true = np.array(self.inference_y)
                if len(y_true.shape) != 1:
                    y_true = np.array([np.argmax(yt) for yt in y_true])

                self.plot_confusion_matrix(y_true, y_pred, self.class_labels,
                                           self.data_dir + '/CNN_confusion_matrix_{}.png'.format(self.data_set))
                class_report = classification_report(y_true, y_pred, labels=[i for i in range(0, self.num_classes)],
                                                     target_names=self.class_labels)
                with open(self.data_dir + "/classification_report_{}.txt".format(self.data_set), 'w') as f:
                    f.write('RF accuracy score on test set is {:.2f}\n\n'.format(
                        accuracy_score(y_true, y_pred) * 100))
                    f.write(class_report)
                print(class_report)


    def plot_confusion_matrix(self, y_true, y_pred, classes, save_path, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        :param y_true:      groundtruth labels
        :param y_pred:      predicted labels
        :param classes:     class names (strings) for plotting as tick labels
        :param normalize:   normalize confusion matrix True/False
        :param cmap:        colormat to use
        :return:
        """

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
            cm_title = 'Normalized confusion matrix'
        else:
            print('Confusion matrix, without normalization')
            cm_title = 'Confusion matrix'

        title = cm_title + ", Accuracy: {0: 0.1f} % ".format(accuracy_score(y_true, y_pred) * 100)

        print(cm)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig = plt.gcf()
        fig.savefig(save_path)
        plt.close(fig)

        return cm


    # def _run_tensorflow_pipeline_dep(self):
    #     tf.reset_default_graph()
    #     if self.chpnt2load:
    #         try:
    #             latest_ckpt = tf.train.latest_checkpoint(self.chpnt2load)
    #         except:
    #             raise ValueError('Mo such checkpoint is found ' + self.chpnt2load)
    #     else:
    #         raise ValueError("No checkpoint provided")
    #         # model_file_path = os.path.join('experiments', 'ckpnt_logs', self.data_set, self.timestamp,
    #         #                                self.timestamp + '_split_' + str(split))
    #
    #     saver = tf.train.import_meta_graph(latest_ckpt + '.meta')
    #     graph = tf.get_default_graph()
    #     if self.gpu_load != 0:
    #         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_load)
    #         config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    #         config.gpu_options.allow_growth = True
    #     else:
    #         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #         config = tf.ConfigProto(device_count={'CPU': 1})
    #
    #     # h5_file = h5py.File(self.h5_data_file, 'r')
    #
    #     with tf.Session(graph=graph, config=config) as sess:
    #         saver.restore(sess, latest_ckpt)
    #
    #         X = tf.get_collection("inputs_train")[0]
    #         pred = tf.get_collection("outputs_train")[0]
    #         mode = tf.get_collection_ref("mode_train")[0]
    #         y_true = tf.get_collection("y_true")
    #         # if self.gradcam_record:
    #         #     y_true = tf.get_collection("y_true")[0]
    #         #     gradcam_in_data = tf.get_collection("gradcam_in_data")[0]
    #         #     gradcam_gb_grads = tf.get_collection("gradcam_gb_grads")[0]
    #         #     gradcam_conv_outs = []
    #         #     gradcam_first_derivatives = []
    #         #     gradcam_second_derivatives = []
    #         #     gradcam_third_derivatives = []
    #         #     for i in range(self.gradcam_layers):
    #         #         gradcam_conv_outs.append(tf.get_collection("gradcam_conv_outs_{}".format(i))[0])
    #         #         gradcam_first_derivatives.append(tf.get_collection("gradcam_first_derivatives_{}".format(i))[0])
    #         #         gradcam_second_derivatives.append(tf.get_collection("gradcam_second_derivatives_{}".format(i))[0])
    #         #         gradcam_third_derivatives.append(tf.get_collection("gradcam_third_derivatives_{}".format(i))[0])
    #         #     gradcam_op = [gradcam_in_data, gradcam_gb_grads, gradcam_conv_outs, gradcam_first_derivatives,
    #         #                   gradcam_second_derivatives, gradcam_third_derivatives]
    #         self.pred_result = []
    #         # gradcam_results = []
    #
    #         # test_generator = BatchIterator(h5_file, 'test', self.img_size, self.num_classes, 1, self.task_type)
    #         # test_lim = test_generator.get_max_lim()
    #
    #         count = 0
    #         for i in tqdm(range(0, len(self.inference_X)), total=len(self.inference_X), unit=' steps', desc='Inference'):
    #             start_time = time.time()
    #             # ff = list(i)
    #             if self.task_type == 'classification':
    #                 pred_result_ = sess.run(pred, feed_dict={
    #                     X: np.reshape(self.inference_X[i], [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False})
    #                 self.pred_result.extend(pred_result_)
    #                 # Calculate Grad-CAM parameters
    #                 # if self.gradcam_record:
    #                 #     gradcam_params = sess.run(gradcam_op, feed_dict={
    #                 #         X: np.reshape(self.inference_X[i], [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False, y_true: pred_result_})
    #                 #     # pred_result.extend(pred_result_)
    #                 #     gradcam_results_ = []
    #                 #     # Run Grad-CAM
    #                 #     for cl in range(self.gradcam_layers):
    #                 #         gradcam_results_.append((gradcam(gradcam_params, cl), gradcam_params[0]))
    #                 #     count = count + 1
    #                 #     # Save Grad-CAM results
    #                 #     save_gradcam(gradcam_results_, self.data_dir, count, self.gradcam_layers_max)
    #                 #     pred_result = sess.run(pred, feed_dict={X: [np.reshape(f[0], [self.img_size[0], self.img_size[1], self.img_size[2]]) for f
    #                 #                         in ff], mode: False})
    #
    #
    #
    #
    #             elif self.task_type == 'segmentation':
    #                 # if self.y:
    #                 mask_pred = sess.run(pred, feed_dict={
    #                     X: np.reshape(self.inference_X[i], [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False})
    #
    #                 #
    #                 # for fr in ff:
    #                 #     <<<<<<< .mine
    #                 #     mask_pred = sess.run(pred, feed_dict={X: np.reshape(fr[0], [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False})
    #                 #     mask = tf.nn.sigmoid(mask_pred).eval()
    #                 #     m_r = np.argmax(mask, axis=3)
    #                 #     f, (ax1, ax2) = plt.subplots(1, 2)
    #                 #     # f_r = np.argmax(fr[1], axis=2)
    #                 #     ax2.imshow(fr[0])
    #                 #     ax1.imshow(np.reshape(m_r, [self.img_size[0], self.img_size[1]]))
    #                 #     # ax1.imshow(np.reshape(f_r, [self.img_size[0], self.img_size[1]]))
    #                 #     f.savefig(os.path.join(self.data_dir, 'results', str(count) + '_.png'))  # save the figure to file
    #                 #     plt.close(f)
    #                 #     plt.show()
    #                 #
    #                 #                             mask_pred = sess.run(pred, feed_dict={X: np.reshape(fr[0], [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False})
    #                 #     ||||||| .r165
    #                 #     mask_pred = sess.run(pred, feed_dict={
    #                 #         X: np.reshape(fr[0], [1, self.img_size[0], self.img_size[1], self.img_size[2]]),
    #                 #         mode: False})
    #                 #     =======
    #                 #                             mask_pred, gradcam_params_ = sess.run([pred, gradcam_op], feed_dict={X: np.reshape(fr[0], [1, self.img_size[0], self.img_size[1], self.img_size[2]]), mode: False})
    #                 #     >>>>>>> .r281
    #                 #                             mask = tf.nn.sigmoid(mask_pred).eval()
    #                 #                             m_r = np.argmax(mask, axis=3)
    #                 #                             f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #                 #                             f_r = np.argmax(fr[1], axis=2)
    #                 #                             ax3.imshow(fr[0])
    #                 #                             ax3.set(xlabel='original mask')
    #                 #                             ax2.imshow(np.reshape(m_r, [self.img_size[0], self.img_size[1]]))
    #                 #                             ax2.set(xlabel='generated segmentation')
    #                 #                             ax1.imshow(np.reshape(f_r, [self.img_size[0], self.img_size[1]]))
    #                 #                             ax1.set(xlabel='original image')
    #                 #                             f.savefig(os.path.join(self.data_dir, 'results', str(count) + '_.png'))  # save the figure to file
    #                 #                             plt.close(f)
    #                 #
    #                 #     mask = tf.nn.sigmoid(mask_pred).eval()
    #                 #     m_r = np.argmax(mask, axis=3)
    #                 #     f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #                 #     # f_r = np.argmax(fr[1], axis=2)
    #                 #     ax3.imshow(fr[0])
    #                 #     ax3.set(xlabel='original mask')
    #                 #     ax2.imshow(np.reshape(m_r, [self.img_size[0], self.img_size[1]]))
    #                 #     # ax2.set(xlabel='generated segmentation')
    #                 #     # ax1.imshow(np.reshape(f_r, [self.img_size[0], self.img_size[1]]))
    #                 #     ax1.set(xlabel='original image')
    #                 #     f.savefig(os.path.join(self.data_dir, 'results', str(count) + '_.png'))  # save the figure to file
    #                 #     plt.close(f)
    #
    #                 mask = tf.nn.sigmoid(mask_pred).eval()
    #                 m_r = np.argmax(mask, axis=3)
    #                 f, (ax1, ax2) = plt.subplots(1, 2)
    #                 # f_r = np.argmax(fr[1], axis=2)
    #
    #                 ax2.imshow(np.reshape(m_r, [self.img_size[0], self.img_size[1]]))
    #                 ax2.set(xlabel='generated segmentation')
    #                 ax1.imshow(np.reshape(self.inference_X[i], [self.img_size[0], self.img_size[1]]))
    #                 ax1.set(xlabel='original image')
    #                 f.savefig(os.path.join(self.data_dir, 'results', str(count) + '_.png'), figsize=(24, 18),
    #                           dpi=600)  # save the figure to file
    #                 plt.close(f)
    #
    #                 count = count + 1
    #
    #         if self.task_type == 'classification':
    #
    #             if self.loss_type == 'mse':
    #                 m = 0
    #                 y_pred = np.array(self.pred_result)
    #                 y_true = np.array(self.inference_y)
    #                 for i in range(0, len(y_pred)):
    #                     mse_test = np.sqrt(np.sum(((y_true[i] - y_pred[i]) ** 2) / 2))
    #                     # mse_test = sess.run(mse_loss(y_true[i], y_pred[i]))
    #                     m += mse_test
    #                     print("Prediction: {} GT: {} Score {}%".format(str(y_pred[i]), str(y_true[i]),
    #                                                                    int(mse_test * 100)))
    #
    #                 # mse_test = np.mean(np.sqrt(np.sum(((y_true - y_pred) ** 2)/2)))
    #                 print("Overall: {}%".format(int(m / len(y_pred) * 100)))
    #             else:
    #                 y_pred = np.argmax(np.array(self.pred_result).squeeze(), axis=1)
    #                 # one hot to label encoding - 1 = FRAUD, 0 = OK
    #                 # y_train = [1 if np.all(yt == np.array([0, 1])) else 0 for yt in y_train]
    #                 y_true = np.array(self.inference_y)
    #                 if len(y_true.shape) != 1:
    #                     y_true = np.array([np.argmax(yt) for yt in y_true])
    #
    #                 self.plot_confusion_matrix(y_true, y_pred, self.class_labels,
    #                                            self.data_dir + '/CNN_confusion_matrix_{}.png'.format(self.data_set))
    #                 class_report = classification_report(y_true, y_pred, labels=[i for i in range(0, self.num_classes)],
    #                                                      target_names=self.class_labels)
    #                 with open(self.data_dir + "/classification_report_{}.txt".format(self.data_set), 'w') as f:
    #                     f.write('RF accuracy score on test set is {:.2f}\n\n'.format(
    #                         accuracy_score(y_true, y_pred) * 100))
    #                     f.write(class_report)
    #                 print(class_report)
