#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 02/21/2018 16:55 $
# by : shepeleva $
# SVN : $
#

# --- imports -----------------------------------------------------------------

import numpy as np
from utils.utils import *


class BatchIterator:

    def __init__(self, file_name, data_split, img_size, num_lbls, batch_size, task_type, is_training, augment_dict=None,
                 shuffle_key=False, colormap=None):
        self.file_name = file_name
        self.data_split = data_split
        self.task_type = task_type
        self.is_training = is_training
        self.augment_dict = augment_dict
        self._preprocess_data(img_size, num_lbls, batch_size, colormap)


        if shuffle_key and not self.is_training:
            self._shuffle()
        else:
            self.permutation_list = self.data_split

        if (self.task_type == 'classification') or (self.task_type == 'segmentation'):
            self.iterator = iter(self.permutation_list[0:self.max_lim])
        elif self.task_type == 'prediction':

            frames_max = [int(self.max_num_frames[i] / self.batch_size)*self.batch_size for i in self.permutation_list]
            scenes_list_g = sum([self.permutation_list[i] * frames_max[i] for i in range(len(frames_max))], [])
            frames_list_g = [k for i in frames_max for k in range(i)]

            scenes_list_x = scenes_list_g[self.batch_size-1::self.batch_size]
            frames_list_x = frames_list_g[self.batch_size-1::self.batch_size]
            scenes_list_y = scenes_list_g[::self.batch_size]
            frames_list_y = frames_list_g[::self.batch_size]

            self.iterator = iter(zip(scenes_list_x, frames_list_x, scenes_list_y, frames_list_y))
        elif self.task_type == 'detection':
            raise ValueError("Not implemented")

        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.on_going = False
        else:
            self.on_going = True

    def _preprocess_data(self, img_size, num_lbls, batch_size, colormap):
        self.X_key = 'X_data'
        self.y_key = 'y_data'
        self.size = len(self.data_split)
        self.img_size = img_size
        self.num_lbls = num_lbls
        self.batch_size = batch_size
        if self.is_training:
            self.max_lim = int(self.size / self.batch_size) * self.batch_size
        else:
            self.max_lim = self.size
        if self.task_type is 'segmentation':
            self.colormap = colormap



    def get_max_lim(self):
        return int(self.max_lim / self.batch_size)

    def _shuffle(self):
        import random
        self.permutation_list = random.sample(self.data_split, len(self.data_split))

    def __iter__(self):
        if self.task_type == 'classification':
            while self.on_going:
                yield self._next_batch_classification()
        elif self.task_type == 'segmentation':
            while self.on_going:
                yield self._next_batch_segmentation()
        elif self.task_type == 'prediction':
            while self.on_going:
                yield self._next_batch_prediction()


    # @jit(nopython=True)
    def _next_batch_segmentation(self):
        imgs = self.file_name[self.X_key][self.current]
        if self.file_name[self.y_key][:].size != 0:
            msks = self.file_name[self.y_key][self.current]
            if not list(self.img_size) == imgs[0].shape:
                imgs, msks = self._resize_data(imgs, msks)
            if self.augment_dict:
                imgs, msks = self._augment_data(imgs, msks)
            msks = self._recode_masks(msks)
            yield imgs, msks
        else:
            if not list(self.img_size) == imgs[0].shape:
                imgs = self._resize_data(imgs)
            if self.augment_dict:
                imgs = self._augment_data(imgs)
            yield imgs, None
        # start enumerate at 1 because we already yielded the last saved item
        for num, item in enumerate(self.iterator, 1):
            imgs = self.file_name[self.X_key][item]
            if self.file_name[self.y_key][:].size != 0:
                msks = self.file_name[self.y_key][item]
                if not list(self.img_size) == imgs[0].shape:
                    imgs, msks = self._resize_data(imgs, msks)
                if self.augment_dict:
                    imgs, msks = self._augment_data(imgs, msks)
                self.current = item
                if num == self.batch_size:
                    break
                msks = self._recode_masks(msks)
                yield imgs, msks
            else:
                if not list(self.img_size) == imgs[0].shape:
                    imgs = self._resize_data(imgs)
                if self.augment_dict:
                    imgs = self._augment_data(imgs)
                self.current = item
                if num == self.batch_size:
                    break
                yield imgs, None
        else:
            self.on_going = False

    # @jit(nopython=True)
    def _next_batch_classification(self):
        imgs = self.file_name[self.X_key][self.current]
        lbls = self.file_name[self.y_key][self.current]
        if not list(self.img_size) == imgs[0].shape:
            imgs = self._resize_data(imgs)
        if self.augment_dict:
            imgs = self._augment_data(imgs)
        yield imgs, lbls
        for num, item in enumerate(self.iterator, 1):
            imgs = self.file_name[self.X_key][item]
            lbls = self.file_name[self.y_key][item]
            if not list(self.img_size) == imgs[0].shape:
                imgs = self._resize_data(imgs)
            if self.augment_dict:
                imgs = self._augment_data(imgs)
            self.current = item
            if num == self.batch_size:
                break
            yield imgs, lbls
        else:
            self.on_going = False

    # @jit(nopython=True)
    def _next_batch_prediction(self):
        x_ped_id = np.where(self.file_name[self.X_key][self.current[0]][0] == self.current[1])[0]
        y_ped_id = np.where(self.file_name[self.X_key][self.current[2]][0] == self.current[3])[0]
        x = self.file_name[self.X_key][self.current[0]][2::, x_ped_id[0]:x_ped_id[-1]]
        y = self.file_name[self.X_key][self.current[2]][2::, y_ped_id[0]:y_ped_id[-1]]
        yield x, y
        # start enumerate at 1 because we already yielded the last saved item
        for num, item in enumerate(self.iterator, 1):
            x_ped_id = np.where(self.file_name[self.X_key][item[0]][0] == item[1])[0]
            y_ped_id = np.where(self.file_name[self.X_key][item[2]][0] == item[3])[0]
            x = self.file_name[self.X_key][item[0]][2::, x_ped_id[0]:x_ped_id[-1]]
            y = self.file_name[self.X_key][item[2]][2::, y_ped_id[0]:y_ped_id[-1]]
            self.current = item
            if num == self.batch_size:
                break
            yield x, y
        else:
            self.on_going = False

    # @jit(nopython=True)
    def _resize_data(self, imgs, msks=None):
        """

        :param imgs:
        :param msks:
        :return:
        """
        if msks is None:
            imgs = resize_images_cpu(imgs, self.img_size)
            # imgs = cv2.resize(imgs, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            return imgs
        else:
            imgs = resize_images_cpu(imgs, self.img_size)
            msks = resize_images_cpu(msks, self.img_size)
            # imgs = cv2.resize(imgs, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            # msks = cv2.resize(msks, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            return imgs, msks

    # @jit(nopython=True)
    def _augment_data(self, imgs, msks=None):
        """

        :param n:
        :param imgs:
        :param msks:
        :return:
        """
        if msks is None:
            if self.augment_dict['flip_hor']:
                if np.random.randint(2) == 1:
                    imgs = np.flip(imgs, axis=0)
            if self.augment_dict['flip_vert']:
                if np.random.randint(2) == 1:
                    imgs = np.flip(imgs, axis=1)
            return imgs
        else:
            if self.augment_dict['flip_hor']:
                if np.random.randint(2) == 1:
                    imgs = np.flip(imgs, axis=0)
                    msks = np.flip(msks, axis=0)
            if self.augment_dict['flip_vert']:
                if np.random.randint(2) == 1:
                    imgs = np.flip(imgs, axis=1)
                    msks = np.flip(msks, axis=1)
            return imgs, msks

    # @jit(nopython=True)
    def _recode_masks(self, msks):
        # only if segmentation colors are encoded as class labels or binary
        if ((len(msks.shape) == 3) and (np.array_equal(msks[:, :, 0], msks[:, :, 1]))) or (len(msks.shape) != 3):
            new_img = np.zeros([msks.shape[0], msks.shape[1], self.num_lbls])
            img = msks[:, :, 0]
            # binary segmentation - known bag: binary segmentation only
            if self.num_lbls == 2:
                img = img//255
                new_img[:, :, 0] = img
                new_img[:, :, 1] = 1 - img
            else:
                for j in range(0, self.num_lbls):
                    x_ind, y_ind = np.where(img == j)
                    for ind in range(0, len(x_ind)):
                        new_img[x_ind[[ind]], y_ind[ind], j] = 1
            return new_img
        else:  # if segmentation colors are encoded as rgb
            if self.colormap and isinstance(self.colormap, dict) and (len(self.colormap) == self.num_lbls):
                new_img = np.zeros([msks.shape[0], msks.shape[1], self.num_lbls])
                for k, v in self.colormap:
                    x_ind, y_ind = np.where(
                        (msks[:, :, 0] == v[0]) & (msks[:, :, 1] == v[1]) & (msks[:, :, 2] == v[2]))
                    for ind in range(0, len(x_ind)):
                        new_img[x_ind[[ind]], y_ind[ind], k] = 1
                return new_img
            else:
                raise ValueError("Colormap is required in a dictionary format {label: (R, G, B}")