from multiprocessing import Pool, Queue
import numpy as np
import os
import cv2
import random

import cv2


image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])


def read_info(data_path):
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')
    data_info = []
    for img_name in os.listdir(images_dir):
        raw_name = os.path.splitext(img_name)[0]
        gt_path = os.path.join(labels_dir, raw_name + '.regions.txt')
        img_path = os.path.join(images_dir, img_name)
        assert os.path.isfile(img_path)
        assert os.path.isfile(gt_path)
        data_info.append([img_path, gt_path])
    print('%i images' % len(data_info))
    return data_info


class ReaderOpts:
    def __init__(self, data_path, batch_size, img_height, img_width, gt_height, gt_width):
        self.data_path = data_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.gt_height = gt_height
        self.gt_width = gt_width


class Reader:
    def __init__(self, opts):
        self.opts = opts
        self.data_info = read_info(opts.data_path)
        self.nbatches = len(self.data_info) // opts.batch_size
        self.batch_idx = 0
        self.num_classes = 8
        random.shuffle(self.data_info)

    def read_item(self):
        image = cv2.imread(self.data_info[self.batch_idx][0])
        original_height, original_width, num_channels = image.shape
        assert num_channels == 3
        image = cv2.resize(image, (self.opts.img_width, self.opts.img_height))
        image = image.astype(np.float32) / 255.0
        image = image - image_means

        with open(self.data_info[self.batch_idx][1], 'r') as fid:
            lines = fid.readlines()

        gt_original_size = np.zeros((original_height, original_width), np.int32)
        assert len(lines) == original_height
        row = 0
        for line in lines:
            line_split = line.split(' ')
            assert len(line_split) == original_width
            for col in range(original_width):
                gt_original_size[row, col] = int(line_split[col])
                if gt_original_size[row, col] < 0:
                    gt_original_size[row, col] = -1
            row += 1

        gt = cv2.resize(gt_original_size, (self.opts.gt_width, self.opts.gt_height), interpolation=cv2.INTER_NEAREST)

        return image, gt

    def get_batch(self):
        batch_imgs = np.zeros((self.opts.batch_size, self.opts.img_height, self.opts.img_width, 3), dtype=np.float32)
        batch_gt = np.zeros((self.opts.batch_size, self.opts.gt_height, self.opts.gt_width), dtype=np.int32)
        for i in range(self.opts.batch_size):
            img, gt = self.read_item()
            batch_imgs[i, :, :, :] = img
            batch_gt[i, :, :] = gt

        self.batch_idx += 1

        if self.batch_idx == self.nbatches:
            random.shuffle(self.data_info)
            self.batch_idx = 0

        return batch_imgs, batch_gt

