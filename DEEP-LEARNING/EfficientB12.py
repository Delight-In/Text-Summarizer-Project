import tensorflow as tf
from tensorflow.keras import layers, models

def swish(x):
    return x * tf.keras.backend.sigmoid(x)

def conv_block(inputs, filters, kernel_size, strides=(1, 1), expansion_factor=1):
    in_channels = inputs.shape[-1]
    x = layers.Conv2D(in_channels * expansion_factor, (1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(swish)(x)

    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(swish)(x)

    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if in_channels == filters and strides == (1, 1):
        return layers.add([inputs, x])
    return x

def EfficientNetB12(input_shape=(380, 380, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(swish)(x)

    # MBConv Blocks
    x = conv_block(x, filters=64, kernel_size=(3, 3), strides=(1, 1), expansion_factor=1)

    x = conv_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2), expansion_factor=6)
    x = conv_block(x, filters=128, kernel_size=(3, 3), strides=(1, 1), expansion_factor=6)

    x = conv_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2), expansion_factor=6)
    x = conv_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1), expansion_factor=6)

    x = conv_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2), expansion_factor=6)
    x = conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1), expansion_factor=6)

    # More MBConv Blocks for B12 (increase depth and width)
    for _ in range(4):
        x = conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1), expansion_factor=6)

    x = conv_block(x, filters=1024, kernel_size=(3, 3), strides=(2, 2), expansion_factor=6)
    x = conv_block(x, filters=1024, kernel_size=(3, 3), strides=(1, 1), expansion_factor=6)

    # Final Layers
    x = layers.Conv2D(1280, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(swish)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Create the EfficientNet B12 model
model = EfficientNetB12()
model.summary()
