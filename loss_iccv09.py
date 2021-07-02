import tensorflow as tf


class LossLayer:
    def __call__(self, y_true, y_pred):
        # y_true: (batch_size, output_height, output_width)
        # y_pred: (batch_size, output_height, output_width, nclasses)
        mask = tf.cast(y_true >= 0, tf.float32)
        y_true = tf.maximum(y_true, 0)  # To avoid negative values
        # TODO: Any problem below with y_true having negative values?
        loss_pixels = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)\
            (y_true, y_pred)  # (batch_size, output_height, output_width)
        loss_pixels = loss_pixels * mask
        denom = tf.cast(tf.shape(loss_pixels)[0] * tf.shape(loss_pixels)[1] * tf.shape(loss_pixels)[2], tf.float32)
        loss = tf.reduce_sum(loss_pixels) / denom
        return loss

