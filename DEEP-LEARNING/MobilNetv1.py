import tensorflow as tf
from tensorflow.keras import layers, models

def depthwise_separable_conv(x, filters, kernel_size, strides=(1, 1), padding='same'):
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Pointwise convolution
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

def MobileNet(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Depthwise separable convolutions
    x = depthwise_separable_conv(x, 64, (3, 3), strides=(1, 1))

    x = depthwise_separable_conv(x, 128, (3, 3), strides=(2, 2))
    x = depthwise_separable_conv(x, 128, (3, 3), strides=(1, 1))

    x = depthwise_separable_conv(x, 256, (3, 3), strides=(2, 2))
    x = depthwise_separable_conv(x, 256, (3, 3), strides=(1, 1))

    x = depthwise_separable_conv(x, 512, (3, 3), strides=(2, 2))

    # Add depthwise separable convolutions for each of the 5 layers
    for _ in range(5):
        x = depthwise_separable_conv(x, 512, (3, 3))

    # Final layers
    x = depthwise_separable_conv(x, 1024, (3, 3))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Create the MobileNet model
model = MobileNet()
model.summary()
