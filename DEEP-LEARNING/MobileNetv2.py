import tensorflow as tf
from tensorflow.keras import layers, models

def inverted_residual_block(x, filters, expansion, strides, skip_connect=True):
    # Expansion
    in_channels = x.shape[-1]
    x = layers.Conv2D(expansion * in_channels, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Depthwise Convolution
    x = layers.DepthwiseConv2D((3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Projection
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if skip_connect and strides == 1 and in_channels == filters:
        return layers.add([x, x])
    return x

def MobileNetV2(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Inverted Residual Blocks
    x = inverted_residual_block(x, filters=16, expansion=1, strides=1, skip_connect=False)
    
    x = inverted_residual_block(x, filters=24, expansion=6, strides=2)
    x = inverted_residual_block(x, filters=24, expansion=6, strides=1)

    x = inverted_residual_block(x, filters=32, expansion=6, strides=2)
    x = inverted_residual_block(x, filters=32, expansion=6, strides=1)
    x = inverted_residual_block(x, filters=32, expansion=6, strides=1)

    x = inverted_residual_block(x, filters=64, expansion=6, strides=2)
    x = inverted_residual_block(x, filters=64, expansion=6, strides=1)
    x = inverted_residual_block(x, filters=64, expansion=6, strides=1)
    x = inverted_residual_block(x, filters=64, expansion=6, strides=1)

    x = inverted_residual_block(x, filters=96, expansion=6, strides=1)
    x = inverted_residual_block(x, filters=96, expansion=6, strides=1)
    x = inverted_residual_block(x, filters=96, expansion=6, strides=1)

    x = inverted_residual_block(x, filters=160, expansion=6, strides=2)
    x = inverted_residual_block(x, filters=160, expansion=6, strides=1)
    x = inverted_residual_block(x, filters=160, expansion=6, strides=1)

    x = inverted_residual_block(x, filters=320, expansion=6, strides=1)

    # Final Layers
    x = layers.Conv2D(1280, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Create the MobileNet v2 model
model = MobileNetV2()
model.summary()
