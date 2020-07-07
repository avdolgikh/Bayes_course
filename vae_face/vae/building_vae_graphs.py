import tensorflow as tf
import keras
from keras import backend as K



def build_vae_model(input_shape, latent_dim, conv_filters, conv_kernel, dropout, vae_kl_coef, BN, additional_dense_sf):
    x = keras.layers.Input( batch_shape=input_shape )
    
    encoder_output, last_conv_size = create_encoder(x, latent_dim, conv_filters, conv_kernel, dropout, BN, additional_dense_sf)

    get_t_mean = keras.layers.Lambda(lambda h: h[:, :latent_dim], name="t_mean")
    get_t_log_var = keras.layers.Lambda(lambda h: h[:, latent_dim:], name="t_log_var")
    t_mean = get_t_mean(encoder_output)
    t_log_var = get_t_log_var(encoder_output)

    t = keras.layers.Lambda(sampling, name="t")([t_mean, t_log_var])
    decoder = create_decoder(latent_dim, conv_filters, last_conv_size, conv_kernel, dropout, BN, additional_dense_sf)
    x_decoded_mean = decoder(t)
    
    #print(x, x_decoded_mean)

    loss = vae_loss(x, x_decoded_mean, t_mean, t_log_var, lmbd = vae_kl_coef)
    vae = keras.models.Model(inputs=x, outputs=x_decoded_mean, name="VAE")

    learning_rate = 1e-5
    #optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.5, decay=0.001, nesterov=False)
    optimizer = keras.optimizers.Adam(lr=learning_rate) #, beta_1=0.9, beta_2=0.99, amsgrad=False
    #optimizer = keras.optimizers.Adadelta(lr=learning_rate, rho=0.97)
    #optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    # We can use SGD with big lr and big decay
    # or use custom algorithm: increase lr if average loss-delta is small (when loss is big)

    vae.compile(optimizer, loss=lambda x, y: loss)
    #vae.summary()
    #decoder.summary()

    return vae, decoder



def vae_loss(x, x_decoded_mean, t_mean, t_log_var, lmbd = 0.5):
    """Returns the value of negative Variational Lower Bound
    
    The inputs are tf.Tensor
        x: (batch_size x (image_size)) matrix with images
        x_decoded_mean: (batch_size x (image_size)) mean of the distribution p(x | t)
        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)
    
    Returns:
        A tf.Tensor with one element (averaged across the batch)
    
    minimize (MSE + KL)

    mean( || x - x_decoded_mean ||^2  ) + 
    -0.5 * mean( t_log_var - exp(t_log_var) - t_mean**2 + 1 )

    lmbd = 0.5
    Use own value as hyper-param
    """

    #x = tf.reshape(x, [x.shape[0], -1])
    #x_decoded_mean = tf.reshape(x_decoded_mean, [x_decoded_mean.shape[0], -1])

    mse = tf.reduce_mean( tf.squared_difference(x, x_decoded_mean))

    # KL divergence b/w q(t) and standard normal distribution:
    kl = -lmbd * tf.reduce_mean( t_log_var - tf.exp(t_log_var) - tf.square(t_mean) + 1 )

    return (mse + kl)

def sampling(args):
    """Returns sample from a distribution N(args[0], diag(args[1]))
    
    The sample should be computed with reparametrization trick.
    
    The inputs are tf.Tensor
        args[0]: (batch_size x latent_dim) mean of the desired distribution
        args[1]: (batch_size x latent_dim) logarithm of the variance vector of the desired distribution
    
    Returns:
        A tf.Tensor of size (batch_size x latent_dim), the samples.
    """
    t_mean, t_log_var = args
    eps = tf.random.normal(t_mean.shape, 0, 1, dtype=tf.float32)
    
    return t_mean + eps * tf.sqrt( tf.exp(t_log_var) )

def conv_layer(layer, filters, kernel, activation, stride, dropout, BN):
    layer = keras.layers.Conv2D(filters=filters,
                                kernel_size=(kernel, kernel),
                                strides=(stride,stride),
                                activation=None,
                                padding="same", # "same", "valid"
                                kernel_initializer='he_normal' #'TruncatedNormal' #'he_normal' # 'he_uniform'
                                ,use_bias=False
                               )(layer)
    if BN:
        layer = keras.layers.BatchNormalization(axis=-1)(layer)
    layer = keras.layers.Activation(activation)(layer)
    if dropout is not None:
        layer = keras.layers.Dropout(dropout)(layer)
    return layer

def dense_layer(layer, n_units, activation, dropout, BN):
    layer = keras.layers.Dense(n_units, activation=None, use_bias=True
                               ,kernel_initializer='he_uniform' #'TruncatedNormal' #'he_normal' # 'he_uniform'
                              )(layer)
    if BN:
        layer = keras.layers.BatchNormalization(axis=-1)(layer)
    layer = keras.layers.Activation(activation)(layer)
    if dropout is not None:
        layer = keras.layers.Dropout(dropout)(layer)
    return layer

def create_encoder(input, latent_dim, filters, kernel, dropout, BN, additional_dense_sf):
    layer = input
    for filter in filters:        
        layer = conv_layer(layer, filters=filter, kernel=kernel, activation="selu", stride=1, dropout=dropout, BN=BN)
        layer = conv_layer(layer, filters=filter, kernel=kernel, activation="selu", stride=2, dropout=dropout, BN=BN)

    last_conv_size = (int(str( layer.shape[1] )), int(str( layer.shape[2] )))
    print(last_conv_size)

    layer = keras.layers.Flatten()(layer)
    layer = dense_layer(layer, additional_dense_sf * latent_dim, "selu", dropout, BN)
    layer = dense_layer(layer, 2 * latent_dim, "linear", dropout, BN)
    return layer, last_conv_size

def create_decoder(input_dim, filters, first_conv_size, kernel, dropout, BN, additional_dense_sf):
    decoder = keras.models.Sequential(name='Decoder')
    decoder.add( keras.layers.InputLayer([input_dim]) )

    first_conv_size_1, first_conv_size_2 = first_conv_size
    
    decoder.add( keras.layers.Dense( int( 2./additional_dense_sf * first_conv_size_1 * first_conv_size_2 * filters[-1]) ) )
    if BN:
        decoder.add( keras.layers.BatchNormalization(axis=-1) )
    decoder.add( keras.layers.Activation("selu") )
    if dropout is not None:        
        decoder.add( keras.layers.Dropout(dropout) )

    decoder.add( keras.layers.Dense(first_conv_size_1 * first_conv_size_2 * filters[-1] ) )
    if BN:
        decoder.add( keras.layers.BatchNormalization(axis=-1) )
    decoder.add( keras.layers.Activation("selu") )
    if dropout is not None:        
        decoder.add( keras.layers.Dropout(dropout) )    

    decoder.add( keras.layers.Reshape((first_conv_size_1, first_conv_size_2, filters[-1])) )

    def add_deconv_block(filter, stride):
        decoder.add( keras.layers.Conv2DTranspose(filters=filter, kernel_size=kernel, strides=stride, padding='same', use_bias=False) )
        if BN:
            decoder.add( keras.layers.BatchNormalization(axis=-1) )
        decoder.add( layer = keras.layers.Activation("selu") )
        if dropout is not None:
            decoder.add( keras.layers.Dropout(dropout) )

    for filter in reversed(filters[1:]):
        add_deconv_block(filter, stride=2)
        add_deconv_block(filter, stride=1)

    decoder.add( keras.layers.Conv2DTranspose(  filters=3, kernel_size=kernel, strides=2, padding='same'
                                                #,activation="linear"
                                                #,activity_regularizer = keras.regularizers.l2(1e-3)
                                            ))
    return decoder


