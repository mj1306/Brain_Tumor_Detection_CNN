from layers import upsample, downsample
import tensorflow as tf

def WNet (input_size=(128, 128, 3), n_filters=32, n_classes=3):

    inputs = tf.keras.layers.Input(input_size)

    encoded_output_1, cache_1 = upsample(inputs, n_filters)
    decoded_output_1 = downsample(encoded_output_1, cache_1, n_filters)
    encoded_output_2, cache_2 = upsample(decoded_output_1, n_filters)
    decoded_output_2 = downsample(encoded_output_2, cache_2, n_filters)

    conv9 = tf.keras.layers.Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(decoded_output_2)

    conv10 = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
