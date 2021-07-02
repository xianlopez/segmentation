import cv2
import numpy as np

import voc_classes
from reader_voc import image_means


class_to_color_map = {}
for key in voc_classes.colors_map:
    class_to_color_map[voc_classes.colors_map[key]] = key
class_to_color_map[-1] = voc_classes.color_unknown


def display_first_element(batch_imgs, batch_gt, net_output):
    # batch_imgs: (batch_size, img_height, img_width, 3)
    # batch_gt: (batch_size, output_height, output_width)
    # net_output: (batch_size, output_height, output_width, nclasses)
    img = batch_imgs[0, :, :, :]
    gt = batch_gt[0, :, :]
    pred = net_output[0, :, :, :]

    img += image_means


    # print('pred.shape = ' + str(pred.shape))

    pred_labels = np.argmax(pred, axis=-1)
    # num_background = np.sum(pred_labels == 0)
    # num_circle = np.sum(pred_labels == 1)
    # print('num_background = ' + str(num_background))
    # print('num_circle = ' + str(num_circle))
    # for i in range(8):
    #     num_this_class = int(np.sum(pred_labels == i))
    #     print('num class %i: %i' % (i, num_this_class))

    # for i in range(8):
    #     num_this_class = int(np.sum(gt == i))
    #     print('num gt class %i: %i' % (i, num_this_class))
    # print('num gt unknown: %i' % int(np.sum(gt < 0)))

    output_height, output_width = gt.shape
    gt_color = np.zeros((output_height, output_width, 3), np.uint8)
    pred_color = np.zeros((output_height, output_width, 3), np.uint8)
    for i in range(output_height):
        for j in range(output_width):
            gt_color[i, j, :] = class_to_color_map[gt[i, j]]
            pred_color[i, j, :] = class_to_color_map[np.argmax(pred[i, j, :])]

    gt_color = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
    pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

    cv2.imshow('img', img)
    cv2.imshow('gt', gt_color)
    cv2.imshow('pred', pred_color)
    cv2.waitKey(10)





