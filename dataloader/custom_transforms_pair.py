import torch
import torch.utils.data as data
import torch.nn as nn

import os, math, random
from os.path import *

import cv2
from scipy.misc import imread, imresize, imsave
import imgaug as ia
from imgaug import augmenters as iaa

import numpy as np
from glob import glob

from tools.utils import *


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    #j = int((j+1)/2)+1
    j = int((j+1)/2)
    #j = int(np.ceil((j+1)/2.0))
    #j = int((j+1)/2)
    return j

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.Upsample(size=(size, size), mode='bilinear')
    labelVar = torch.from_numpy(label.transpose(3, 2, 0, 1))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>0.3]  = 1
    label_resized[label_resized != 0]  = 1

    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def aug_pair(img_template, img_search, gt_template, gt_search):

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes2 = lambda aug: iaa.Sometimes(0.9, aug)

    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (2**(-1/8), 2**(1/8)), "y": (2**(-1/8), 2**(1/8))},
                translate_px={"x": (-8, 8), "y": (-8, 8)}, # translate by -20 to +20 percent (per axis)
                cval=(0, 0), # if mode is constant, use a cval between 0 and 255
                mode='edge' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.Add((-10, 10), per_channel=0.5),
        ], random_order=True
        )
    
    seq2 = iaa.Sequential(
        [
            sometimes2(iaa.Affine(
                scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
                rotate=(-5, 5), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=0, # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            #sometimes2(iaa.CoarseDropout(0.2, size_percent=(0.1, 0.5)
            #))
        ], random_order=True
        )

    scale=1
    dim = int(scale*328)

    flip_p = random.uniform(0, 1)

    img_template= flip(img_template,flip_p)
    gt_template= flip(gt_template,flip_p)
    img_search= flip(img_search,flip_p)
    gt_search= flip(gt_search,flip_p)

    # process template
    bb = cv2.boundingRect(gt_template)
    if bb[2] != 0 and bb[3] != 0:
        template = crop_and_padding(img_template, gt_template, (dim, dim))
        t_h, t_w, _ = img_template.shape
        #fg = np.ones([img_template.shape[1], img_template.shape[1], 1]) * 100
        template_mask = crop_and_padding(gt_template, gt_template, (dim, dim))
        bb = cv2.boundingRect(template_mask)
        template_mask= np.zeros([dim, dim, 1])
        if bb[2] != 0 and bb[3] != 0:
            template_mask[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1
    else:
        template = np.zeros([dim, dim, 3]).astype('uint8')
        template_mask = np.zeros([dim, dim, 1])

    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB).astype(float)/255.
    
    # Augment Search Image
    img_search_o = crop_and_padding(img_search, gt_search, (dim, dim)).astype('uint8')
    gt_search_o = crop_and_padding(gt_search, gt_search, (dim, dim)).astype('uint8')

    for i in range(10):
        seq_det = seq.to_deterministic()
        img_search = seq_det.augment_image(img_search_o)
        img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB).astype(float)/255.

        gt_search = ia.SegmentationMapOnImage(gt_search_o, shape=gt_search_o.shape, nb_classes=2)
        gt_search_map = seq_det.augment_segmentation_maps([gt_search])[0]
        gt_search = gt_search_map.get_arr_int()
        bb = cv2.boundingRect(gt_search.astype('uint8'))
        if bb[2] > 10 and bb[3] > 10:
            break
    else:
        img_search = img_search_o
        gt_search = gt_search_o
        
    kernel = np.ones((int(scale*5), int(scale*5)), np.uint8)
    for i in range(10):
        mask = seq2.augment_segmentation_maps([gt_search_map])[0].get_arr_int().astype(float)
        bb = cv2.boundingRect(gt_search.astype('uint8'))
        if bb[2] > 10 and bb[3] > 10:
            break
    else:
        mask = gt_search.copy()
 
    fc = np.zeros([dim, dim, 1])
    if flip_p <= 0.8:
        aug_p = random.uniform(0, 1)
        it = random.randint(1, 3)

        aug = np.expand_dims(cv2.dilate(mask, kernel, iterations=it), 2)
        fc[np.where(aug==1)] = 1
    else:
        fc[:, :, 0] =mask 

    gt_search= np.expand_dims(gt_search, 2)
    gt = np.expand_dims(gt_search, 3)
    label = resize_label_batch(gt.astype(float), dim//2)
    label = label.squeeze(3)


    return img_search, fc, template, template_mask, label


def aug_mask_nodeform(img_template, img_search, gt_template, gt_search, p_mask):

    sometimes = lambda aug: iaa.Sometimes(0.8, aug)

    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (2**(-1/8), 2**(1/8)), "y": (2**(-1/8), 2**(1/8))},
                translate_px={"x": (-8, 8), "y": (-8, 8)}, # translate by -20 to +20 percent (per axis)
                cval=(0, 0), # if mode is constant, use a cval between 0 and 255
                mode='edge' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.Add((-10, 10), per_channel=0.5),
        ], random_order=True
        )
    
    scale=1
    dim = int(scale*328)
    
    # Create Template Image
    flip_p = random.uniform(0, 1)
    img_template= flip(img_template,flip_p)
    gt_template= flip(gt_template,flip_p)

    img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB)
    target = crop_and_padding(img_template, gt_template, (dim, dim))
    mask = crop_and_padding(gt_template, gt_template, (dim, dim))

    seq_det = seq.to_deterministic()
    target = seq_det.augment_image(target).astype(float)/255.
    mask_map = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=2)
    mask= seq_det.augment_segmentation_maps([mask_map])[0].get_arr_int()
    bb = cv2.boundingRect(mask.astype('uint8'))
    template_mask= np.zeros(mask.shape)
    template_mask[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]] = 1

    # Create Search Image
    flip_p = random.uniform(0, 1)
    img_search = flip(img_search,flip_p)
    gt_search = flip(gt_search,flip_p)
    p_mask = flip(p_mask, flip_p)
    
    img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB)
    img_search_o = crop_and_padding(img_search, p_mask, (dim, dim))
    gt_search_o = crop_and_padding(gt_search, p_mask, (dim, dim))
    p_mask_o = crop_and_padding(p_mask, p_mask, (dim, dim))

    if len(np.unique(gt_search_o)) == 1:
        img_search_o = crop_and_padding(img_search, gt_search, (dim, dim))
        gt_search_o = crop_and_padding(gt_search, gt_search, (dim, dim))
        p_mask = gt_search
    for i in range(10):
        seq_det = seq.to_deterministic()
        img_search = seq_det.augment_image(img_search_o).astype(float)/255.

        gt_searchmap = ia.SegmentationMapOnImage(gt_search_o, shape=gt_search.shape, nb_classes=2)
        gt_search = seq_det.augment_segmentation_maps([gt_searchmap])[0].get_arr_int().astype(float)
        mask_map = ia.SegmentationMapOnImage(p_mask_o, shape=p_mask.shape, nb_classes=2)
        p_mask= seq_det.augment_segmentation_maps([mask_map])[0].get_arr_int().astype(float)
        if bb[2] > 10 and bb[3] > 10:
            break

    
    kernel = np.ones((int(scale*3), int(scale*3)), np.uint8)
    it = random.randint(1, 3)
    p_mask = np.expand_dims(cv2.dilate(p_mask.astype(float), kernel, iterations=it), 2)
    template_mask= np.expand_dims(template_mask, 2)

    #image = np.dstack([img_search, p_mask])
    gt_search = np.expand_dims(gt_search, 2)
    gt_search = np.expand_dims(gt_search, 3)
    label = resize_label_batch(gt_search, dim//2)
    label = label.squeeze(3)

    return img_search, fc, template, template_mask, label

