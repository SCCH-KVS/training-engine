# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 30/08/2018 11:06 $
# by : szanto $
# SVN : $
#

# --- imports -----------------------------------------------------------------

import os
import numpy as np
import cv2

np.seterr(divide='ignore', invalid='ignore')

def single_chan_to_three(img):
    """
    Convert a single channel image to 3 channels
    :param img: single channel image
    :return:
    """
    im = img.reshape((img.shape[0], img.shape[1]))
    im3 = np.full((img.shape[0], img.shape[1], 3), 0.0, dtype=np.float32)
    im3[..., 0] = im
    im3[..., 1] = im
    im3[..., 2] = im
    return im3


def grad_cam_plus_plus(params, cl):
    """
    Grad-CAM++ implementation, sources in the readme
    :param params: required inputs for the algorithm
    :param cl: number of the inspected layer
    :return:
    """

    # restructure the given parameters
    images = params[0]
    outputs = params[2][cl]
    gb_vals_ = params[1]
    first_derivatives_ = params[3][cl]
    second_derivatives_ = params[4][cl]
    third_derivatives_ = params[5][cl]

    # return values
    merged_imgs = []
    cams = []
    merged_gb_imgs = []
    gb_grads_vals = []

    # go through the batch of the images
    for i in range(images.shape[0]):
        # get the relevant data for the image
        image = images[i]
        output = outputs[i]
        gb_vals = gb_vals_[i]
        first_derivatives = first_derivatives_[i]
        second_derivatives = second_derivatives_[i]
        third_derivatives = third_derivatives_[i]

        # image parameters width, height channels
        img_w = image.shape[0]
        img_h = image.shape[1]
        img_c = image.shape[2]

        # calculate the sensitivity based on the Grad-CAM++ algorithm
        # magic
        global_sum = np.sum(output.reshape((-1, first_derivatives.shape[2])), axis=0)
        alpha_num = second_derivatives
        alpha_denom = second_derivatives * 2.0 + third_derivatives * global_sum.reshape(
            (1, 1, first_derivatives.shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num / alpha_denom
        weights = np.maximum(first_derivatives, 0.0)
        alphas_threshold = np.where(weights, alphas, 0.0)
        alpha_normalization_constant = np.sum(np.sum(alphas_threshold, axis=0), axis=0)
        alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0,
                                                          alpha_normalization_constant,
                                                          np.ones(alpha_normalization_constant.shape))
        alphas /= alpha_normalization_constant_processed.reshape((1, 1, first_derivatives.shape[2]))
        deep_linearization_weights = np.sum((weights * alphas).reshape((-1, first_derivatives.shape[2])), axis=0)
        grad_CAM_map = np.sum(deep_linearization_weights * output, axis=2)

        # normalisation and upscaling
        eps = 1e-5
        cam = np.maximum(grad_CAM_map, eps)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (img_w, img_h))

        # guided backpropagation
        gb_grads_val = gb_vals
        gb_grads_val -= np.min(gb_grads_val)
        gb_grads_val /= gb_grads_val.max()
        gb_grads_val = gb_grads_val.reshape((img_w, img_h, img_c))

        # convert image to 3chan format if needed
        if img_c == 1:
            gb_grads_val = single_chan_to_three(gb_grads_val)
            image = single_chan_to_three(image)

        # generating the Grad-CAM heatmap
        merged_gb_img = np.dstack((
            gb_grads_val[:, :, 0] * cam,
            gb_grads_val[:, :, 1] * cam,
            gb_grads_val[:, :, 2] * cam,
        ))

        merged_gb_img *= 255
        merged_gb_img = np.clip(merged_gb_img, 0, 255).astype('uint8')

        gb_grads_val *= 255
        gb_grads_val = np.clip(gb_grads_val, 0, 255).astype('uint8')

        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # merging the heatmap and the image
        im = np.uint8(image * 255).reshape(img_w, img_h, 3)
        img = im
        img -= np.min(img)
        img = np.minimum(img, 255)
        merged_img = np.float32(cam) + np.float32(img)
        merged_img = 255 * merged_img / np.max(merged_img)
        merged_img = np.uint8(merged_img)

        merged_imgs.append(merged_img)
        cams.append(cam)
        gb_grads_vals.append(gb_grads_val)
        merged_gb_imgs.append(merged_gb_img)

    return merged_imgs, cams, merged_gb_imgs, gb_grads_vals


def create_folder(path):
    """
    Create a folder if it is not exist
    :param path: path of the folder
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_gradcam_data(data, data_dir, img_num, layers):
    """
    Save data generated with Grad-CAM to files
    :param data: Grad-CAM data in a format: [captured layers: ([[Grad-CAM], [heatmap], [Guided Grad-CAM], [Guided Backprogation]], original image)]
    :param data_dir: path, wehere the files should be saved
    :param img_num: number or id of the image
    :param layers: number of the convolutinal layers in the network
    :return:
    """
    for i in range(len(data)):
        l_idx = layers - i

        [merged_imgs, cams, merged_gb_imgs, gb_grads_vals], original_img = data[i]
        create_folder('{}/gradcam/img_{}'.format(data_dir, img_num))

        cv2.imwrite('{}/gradcam/img_{}/heatmap_{}.jpg'.format(data_dir, img_num, l_idx), cams[0])
        cv2.imwrite('{}/gradcam/img_{}/gradcam_{}.jpg'.format(data_dir, img_num, l_idx), merged_imgs[0])
        cv2.imwrite('{}/gradcam/img_{}/guided_gradcam_{}.jpg'.format(data_dir, img_num, l_idx), merged_gb_imgs[0])
        if i == 0:
            cv2.imwrite('{}/gradcam/img_{}/original.jpg'.format(data_dir, img_num), original_img[0] * 255)
            cv2.imwrite('{}/gradcam/img_{}/guided_backpropagation.jpg'.format(data_dir, img_num), gb_grads_vals[0])
