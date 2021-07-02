import tensorflow as tf
from tensorflow.keras import layers, Model, Input

img_height = 240
img_width = 320


def create_model(nclasses):
    input_tensor = Input(shape=(img_height, img_width, 3))

    filters_dims = [64, 128, 256, 512]

    blocks = []
    x = input_tensor
    for i in range(len(filters_dims)):
        x = layers.Conv2D(filters_dims[i], 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters_dims[i], 3, padding='same', activation='relu')(x)
        blocks.append(x)
        if i < len(filters_dims) - 1:
            x = layers.MaxPool2D()(x)

    for i in range(len(filters_dims) - 2, -1, -1):
        x = tf.keras.layers.UpSampling2D()(x)
        # x = layers.Conv2D(filters_dims[i], 2, padding='same', activation=None)(x)
        x = layers.Conv2D(filters_dims[i], 3, padding='same', activation='relu')(x)
        x = tf.concat([x, blocks[i]], axis=-1)
        x = layers.Conv2D(filters_dims[i], 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters_dims[i], 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(nclasses, 1, padding='same', activation=None)(x)

    return Model(inputs=input_tensor, outputs=x, name='unet')




