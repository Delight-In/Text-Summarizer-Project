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

def Xception(input_shape=(299, 299, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)

    # Entry flow
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = depthwise_separable_conv(x, 128, (3, 3))
    x = depthwise_separable_conv(x, 128, (3, 3))
    
    x = depthwise_separable_conv(x, 256, (3, 3))
    x = depthwise_separable_conv(x, 256, (3, 3))
    
    x = depthwise_separable_conv(x, 728, (3, 3))

    # Middle flow (8 times)
    for _ in range(8):
        residual = x
        x = depthwise_separable_conv(x, 728, (3, 3))
        x = layers.add([x, residual])

    # Exit flow
    x = depthwise_separable_conv(x, 1024, (3, 3))
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Create the Xception model
model = Xception()
model.summary()
