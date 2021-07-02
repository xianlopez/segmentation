import tensorflow as tf


def compute_accuracy(y_true, y_pred):
    # y_true: (batch_size, output_height, output_width)
    # y_pred: (batch_size, output_height, output_width, nclasses)
    mask = tf.cast(y_true >= 0, tf.float32)
    num_known = tf.reduce_sum(mask)
    class_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    # TODO: Shall I add some tolerance to the comparison below (it's in float)?
    hits = tf.cast(class_pred == y_true, tf.float32)
    hits = hits * mask
    accuracy = tf.reduce_sum(hits) / num_known
    return accuracy
