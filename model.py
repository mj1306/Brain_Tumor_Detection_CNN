from layers import upsample, downsample #Functions implemented in layers.py
import tensorflow as tf

def WNet (input_size=(128, 128, 4), n_filters=32, n_classes=3):

    inputs = tf.keras.layers.Input(input_size)

    encoded_output_1, cache_1 = downsample(inputs, n_filters)
    decoded_output_1 = upsample(encoded_output_1, cache_1, n_filters)
    encoded_output_2, cache_2 = downsample(decoded_output_1, n_filters)
    decoded_output_2 = upsample(encoded_output_2, cache_2, n_filters)

    conv9 = tf.keras.layers.Conv3D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(decoded_output_2)

    conv10 = tf.keras.layers.Conv3D(n_classes, 1, padding='same')(conv9)

    output = tf.keras.layers.Softmax()(conv10)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model
