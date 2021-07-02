import tensorflow as tf


class LossLayer:
    def __call__(self, y_true, y_pred):
        # y_true: (batch_size, output_height, output_width)
        # y_pred: (batch_size, output_height, output_width, nclasses)
        known_mask = tf.cast(y_true >= 0, tf.float32)
        background_mask = tf.cast(y_true == 0, tf.float32)
        objects_mask = 1.0 - background_mask
        y_true = tf.maximum(y_true, 0)  # To avoid negative values
        loss_pixels = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)\
            (y_true, y_pred)  # (batch_size, output_height, output_width)
        loss_pixels = loss_pixels * known_mask
        loss_background = loss_pixels * background_mask
        loss_objects = loss_pixels * objects_mask
        denom = tf.cast(tf.shape(y_true)[0] * tf.shape(y_true)[1] * tf.shape(y_true)[2], tf.float32)
        loss = tf.reduce_sum(0.1 * loss_background + loss_objects) / denom
        return loss
