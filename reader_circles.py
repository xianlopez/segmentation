from multiprocessing import Pool, Queue
import numpy as np
import os
import cv2
import random


import sys
sys.path.insert(0, '/home/xian/unet')
from my_unet.datasets import circles

import cv2


image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])


train_dataset, validation_dataset = circles.load_data(100, nx=200, ny=200, splits=(0.8, 0.2))


class ReaderOpts:
    def __init__(self, split, batch_size, img_height, img_width, gt_height, gt_width):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.gt_height = gt_height
        self.gt_width = gt_width
        self.split = split


class Reader:
    def __init__(self, opts):
        self.opts = opts
        self.dataset = train_dataset if opts.split == 'train' else validation_dataset
        self.dataset_iterator = iter(self.dataset)
        self.nbatches = len(self.dataset) // opts.batch_size
        print('self.nbatches = ' + str(self.nbatches))
        self.batch_idx = 0

    def get_batch(self):
        batch_imgs = np.zeros((self.opts.batch_size, self.opts.img_height, self.opts.img_width, 3), dtype=np.float32)
        batch_gt = np.zeros((self.opts.batch_size, self.opts.gt_height, self.opts.gt_width), dtype=np.int32)
        for i in range(self.opts.batch_size):
            item = next(self.dataset_iterator)

            img = item[0].numpy()
            img = cv2.resize(img, (self.opts.img_width, self.opts.img_height))
            img = np.expand_dims(img, axis=-1)
            img = np.tile(img, [1, 1, 3])
            img -= image_means  # TODO: Sure?
            batch_imgs[i, :, :, :] = img

            gt = item[1].numpy()
            gt = np.argmax(gt, axis=-1)
            gt = cv2.resize(gt, (self.opts.gt_width, self.opts.gt_height), interpolation=cv2.INTER_NEAREST)
            batch_gt[i, :, :] = gt

        self.batch_idx += 1

        if self.batch_idx == self.nbatches:
            self.dataset_iterator = iter(self.dataset)
            self.batch_idx = 0

        return batch_imgs, batch_gt

