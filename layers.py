import tensorflow as tf

def encoder_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):

    conv = tf.keras.layers.Conv2D(n_filters, 
                  3,   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    
    conv = tf.keras.layers.Conv2D(n_filters, 
                  3,   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    
    conv = tf.keras.layers.BatchNormalization()(conv, training=False)

    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        output = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        output = conv

    skip_connection = conv

                                    
    return output, skip_connection


def decoder_block(prev_layer_input, skip_layer_input, n_filters=32):

    transpose = tf.keras.layers.Conv2DTranspose(
                 n_filters,
                 (3,3),    
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    residual = tf.keras.layers.concatenate([transpose, skip_layer_input], axis=3)
    
    conv = tf.keras.layers.Conv2D(n_filters, 
                 3,     
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(residual)
    output = tf.keras.layers.Conv2D(n_filters,
                 3,   
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    
    return output
