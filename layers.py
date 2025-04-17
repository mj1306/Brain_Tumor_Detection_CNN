import tensorflow as tf

def encoder_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):

    conv = tf.keras.layers.Conv3D(n_filters, 
                  3,   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    
    conv = tf.keras.layers.Conv3D(n_filters, 
                  3,   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    
    conv = tf.keras.layers.BatchNormalization()(conv, training=False)

    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        output = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2))(conv)    
    else:
        output = conv

    skip_connection = conv

                                    
    return output, skip_connection


def decoder_block(prev_layer_input, skip_layer_input, n_filters=32):

    transpose = tf.keras.layers.Conv3DTranspose(
                 n_filters,
                 (3,3,3),    
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    residual = tf.keras.layers.concatenate([transpose, skip_layer_input], axis=4)
    
    conv = tf.keras.layers.Conv3D(n_filters, 
                 3,     
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(residual)
    output = tf.keras.layers.Conv3D(n_filters,
                 3,   
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    
    return output

def downsample (inputs, n_filters):

    cblock1 = encoder_block(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = encoder_block(cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True)
    cblock3 = encoder_block(cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True)
    cblock4 = encoder_block(cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True)
    encoded_output = encoder_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)
    cache = [cblock1,cblock2,cblock3,cblock4]

    return encoded_output, cache

def upsample(encoded_output, cache, n_filters):

    cblock1,cblock2,cblock3,cblock4 = cache
    ublock6 = decoder_block(encoded_output, cblock4[1],  n_filters * 8)
    ublock7 = decoder_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = decoder_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = decoder_block(ublock8, cblock1[1],  n_filters)

    return ublock9