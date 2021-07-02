from multiprocessing import Pool, Queue
import numpy as np
import os
import cv2
import random

import voc_classes


image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])


def get_items_from_year(voc_path, split, year):
    assert split in ('train', 'val')
    list_name = 'train.txt' if split == 'train' else 'val.txt'
    list_path = os.path.join(voc_path, 'VOC' + str(year), 'ImageSets', 'Segmentation', list_name)
    with open(list_path, 'r') as fid:
        lines = fid.readlines()
    items_paths = []
    file_names = [line.strip() for line in lines]
    for name in file_names:
        img_path = os.path.join(voc_path, 'VOC' + str(year), 'JPEGImages', name + '.jpg')
        gt_path = os.path.join(voc_path, 'VOC' + str(year), 'SegmentationClass', name + '.png')
        items_paths.append([img_path, gt_path])
    print('Year ' + str(year) + ': ' + str(len(items_paths)) + ' items for ' + split)
    return items_paths


def get_items_paths(voc_path, split):
    items_paths = get_items_from_year(voc_path, split, 2007)
    items_paths.extend(get_items_from_year(voc_path, split, 2012))
    print('Total number of items for ' + split + ': ' + str(len(items_paths)))
    return items_paths


def read_batch(batch_info, opts):
    batch_imgs_np = np.zeros((opts.batch_size, opts.img_height, opts.img_width, 3), np.float32)
    batch_gt_np = np.zeros((opts.batch_size, opts.gt_height, opts.gt_width), np.int32)
    for i in range(len(batch_info)):
        item_info = batch_info[i]
        image, gt_mask = read_item(item_info, opts)
        batch_imgs_np[i, :, :, :] = image
        batch_gt_np[i, :, :] = gt_mask
    output_queue.put((batch_imgs_np, batch_gt_np))


def read_item(item_info, opts):
    # Read images:
    image = cv2.imread(item_info[0])
    # Resize:
    image = cv2.resize(image, (opts.img_width, opts.img_height))
    # Make pixel values between 0 and 1:
    image = image.astype(np.float32) / 255.0
    # Subtract mean:
    image = image - image_means

    gt_mask_color = cv2.imread(item_info[1])
    gt_mask_color = cv2.resize(gt_mask_color, (opts.gt_width, opts.gt_height), interpolation=cv2.INTER_NEAREST)
    gt_mask = np.ones((opts.gt_height, opts.gt_width), np.int32) * -1
    for i in range(opts.gt_height):
        for j in range(opts.gt_width):
            color_bgr = gt_mask_color[i, j, :]
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            if color_rgb in voc_classes.colors_map:
                gt_mask[i, j] = voc_classes.colors_map[color_rgb]
            else:
                assert color_rgb == voc_classes.color_unknown

    return image, gt_mask


def init_worker(queue):
    global output_queue
    output_queue = queue


class ReaderOpts:
    def __init__(self, voc_path, split, batch_size, img_height, img_width, gt_height, gt_width, nworkers):
        self.voc_path = voc_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.gt_height = gt_height
        self.gt_width = gt_width
        self.nworkers = nworkers
        self.split = split


class AsyncReader:
    def __init__(self, opts):
        self.opts = opts
        self.data_info = get_items_paths(opts.voc_path, opts.split)
        # self.data_info = self.data_info[:200]
        self.nbatches = len(self.data_info) // opts.batch_size

        self.output_queue = Queue()
        self.pool = Pool(processes=self.opts.nworkers, initializer=init_worker, initargs=(self.output_queue,))
        self.next_batch_idx = 0
        random.shuffle(self.data_info)
        for i in range(min(self.opts.nworkers, self.nbatches)):
            self.add_fetch_task()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        print('Closing AsyncReader')
        self.pool.close()
        # OpenCV seems to play bad with multiprocessing, so I need to add this here. Maybe I could change
        # the reading of the images to use skimage instead of cv2.
        self.pool.terminate()
        self.pool.join()
        print('Closed')

    def add_fetch_task(self):
        batch_info = []
        for i in range(self.opts.batch_size):
            batch_info.append(self.data_info[self.next_batch_idx * self.opts.batch_size + i])
        self.pool.apply_async(read_batch, args=(batch_info, self.opts))
        if self.next_batch_idx == self.nbatches - 1:
            self.next_batch_idx = 0
            random.shuffle(self.data_info)
        else:
            self.next_batch_idx += 1

    def get_batch(self):
        batch_images, batch_gt = self.output_queue.get()
        self.add_fetch_task()
        return batch_images, batch_gt

