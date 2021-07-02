import tensorflow as tf
import numpy as np
from datetime import datetime
from sys import stdout
import shutil
import os

from reader_circles import Reader, ReaderOpts
from loss_voc import LossLayer
from model import create_model
from metrics import compute_accuracy
from display import display_first_element

img_height = 240
img_width = 320

gt_height = 240
gt_width = 320

batch_size = 5
train_reader_opts = ReaderOpts('train', batch_size, img_height, img_width, gt_height, gt_width)
val_reader_opts = ReaderOpts('val', batch_size, img_height, img_width, gt_height, gt_width)

train_reader = Reader(train_reader_opts)
val_reader = Reader(val_reader_opts)

nepochs = 2000

net = create_model(2)
net.summary()
trainable_weights = net.trainable_weights

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

loss_layer = LossLayer()

log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ckpts')


@tf.function
def train_step(imgs, mask_gt, step_count):
    with tf.GradientTape() as tape:
        net_output = net(imgs)
        loss_value = loss_layer(mask_gt, net_output)

    grads = tape.gradient(loss_value, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))

    accuracy = compute_accuracy(mask_gt, net_output)

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step_count)
        tf.summary.scalar('accuracy', accuracy, step=step_count)

    return loss_value, net_output


@tf.function
def eval_step(imgs, mask_gt):
    net_output = net(imgs)
    loss_value = loss_layer(mask_gt, net_output)
    accuracy = compute_accuracy(mask_gt, net_output)
    return loss_value, accuracy


def evaluation_loop(val_reader, step_count):
    print('Evaluating...')
    eval_start = datetime.now()
    accum_loss = 0.0
    accum_accuracy = 0.0
    for batch_idx in range(val_reader.nbatches):
        imgs, mask_gt = val_reader.get_batch()
        imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
        mask_gt_tf = tf.convert_to_tensor(mask_gt, dtype=tf.float32)
        batch_loss, batch_accuracy = eval_step(imgs_tf, mask_gt_tf)
        accum_loss +=batch_loss.numpy()
        accum_accuracy += batch_accuracy.numpy()
    average_loss = accum_loss / float(val_reader.nbatches)
    average_accuracy = accum_accuracy / float(val_reader.nbatches)
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', average_loss, step=step_count)
        tf.summary.scalar('accuracy', average_accuracy, step=step_count)
    print('Evaluation computed in ' + str(datetime.now() - eval_start))


step_count = 0
for epoch in range(nepochs):
    print("\nStart epoch ", epoch + 1)
    epoch_start = datetime.now()
    for batch_idx in range(train_reader.nbatches):
        imgs, mask_gt = train_reader.get_batch()
        step_count_tf = tf.convert_to_tensor(step_count, dtype=tf.int64)
        imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
        mask_gt_tf = tf.convert_to_tensor(mask_gt, dtype=tf.float32)
        loss_value, net_output = train_step(imgs_tf, mask_gt_tf, step_count_tf)
        train_summary_writer.flush()
        stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
        stdout.flush()
        if (batch_idx + 1) % 1 == 0:
            display_first_element(imgs, mask_gt, net_output.numpy())
        step_count += 1
    stdout.write('\n')
    print('Epoch computed in ' + str(datetime.now() - epoch_start))

    evaluation_loop(val_reader, step_count)

    # Save models:
    print('Saving model')
    net.save_weights(os.path.join(save_dir, 'net_' + str(epoch), 'weights'))