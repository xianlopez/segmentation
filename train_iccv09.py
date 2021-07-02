import tensorflow as tf
from datetime import datetime
from sys import stdout
import shutil
import os

from reader_iccv09 import Reader, ReaderOpts
from loss_iccv09 import LossLayer
from model import create_model
from metrics import compute_accuracy
from display import display_first_element

img_height = 240
img_width = 320

gt_height = 240
gt_width = 320

data_path = '/home/xian/iccv09Data'
batch_size = 8
reader_opts = ReaderOpts(data_path, batch_size, img_height, img_width, gt_height, gt_width)

reader = Reader(reader_opts)

nepochs = 200

net = create_model(reader.num_classes)
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


step_count = 0
for epoch in range(nepochs):
    print("\nStart epoch ", epoch + 1)
    epoch_start = datetime.now()
    for batch_idx in range(reader.nbatches):
        imgs, mask_gt = reader.get_batch()
        step_count_tf = tf.convert_to_tensor(step_count, dtype=tf.int64)
        imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
        mask_gt_tf = tf.convert_to_tensor(mask_gt, dtype=tf.float32)
        loss_value, net_output = train_step(imgs_tf, mask_gt_tf, step_count_tf)
        train_summary_writer.flush()
        stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, reader.nbatches, loss_value.numpy()))
        stdout.flush()
        if (batch_idx + 1) % 1 == 0:
            display_first_element(imgs, mask_gt, net_output.numpy())
        step_count += 1
    stdout.write('\n')
    print('Epoch computed in ' + str(datetime.now() - epoch_start))

