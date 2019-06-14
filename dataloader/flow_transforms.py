from __future__ import division
import torch
import random
import numpy as np
import numbers
import pdb
import cv2


class Compose(object):
    """ Composes several co_transforms together.
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target



class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    """

    def __init__(self, size, order=2):
        self.ratio = size
        self.order = order
        if order==0:
            self.code=cv2.INTER_NEAREST
        elif order==1:
            self.code=cv2.INTER_LINEAR
        elif order==2:
            self.code=cv2.INTER_CUBIC

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        ratio = self.ratio

        inputs[0] = cv2.resize(inputs[0], None, fx=ratio,fy=ratio,interpolation=cv2.INTER_CUBIC)
        inputs[1] = cv2.resize(inputs[1], None, fx=ratio,fy=ratio,interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target, None, fx=ratio,fy=ratio,interpolation=self.code) * ratio

        return inputs, target


class RandomCrop(object):
    """ Randomly crop images
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w < tw: tw=w
        if h < th: th=h

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0][y1: y1 + th,x1: x1 + tw]
        inputs[1] = inputs[1][y1: y1 + th,x1: x1 + tw]
        return inputs, target[y1: y1 + th,x1: x1 + tw]


class RandomVdisp(object):
    """Random vertical disparity augmentation
    """

    def __init__(self, angle, px, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle
        self.px = px

    def __call__(self, inputs,target):
        px2 = random.uniform(-self.px,self.px)
        angle2 = random.uniform(-self.angle,self.angle)

        image_center = (np.random.uniform(0,inputs[1].shape[0]),\
                             np.random.uniform(0,inputs[1].shape[1]))
        rot_mat = cv2.getRotationMatrix2D(image_center, angle2, 1.0)
        inputs[1] = cv2.warpAffine(inputs[1], rot_mat, inputs[1].shape[1::-1], flags=cv2.INTER_LINEAR)
        trans_mat = np.float32([[1,0,0],[0,1,px2]])
        inputs[1] = cv2.warpAffine(inputs[1], trans_mat, inputs[1].shape[1::-1], flags=cv2.INTER_LINEAR)
        return inputs,target
