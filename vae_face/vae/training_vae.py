import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (13, 5)
import tensorflow as tf
from keras import backend as K
import gc

from building_vae_graphs import build_vae_model
from stlr_callback import KerasSlantedTriangularLearningRateCallback


FOLDER = '../../experiments/'
TRAIN_SET = '../../experiments/train_celeba.npy'
TEST_SET = '../../experiments/test_celeba.npy'


def load_data(batch_size):
    with open(TRAIN_SET, 'rb') as file:
        x_train = np.load(file)

    x_train = x_train [ : batch_size * int(len(x_train) / batch_size) ]
    x_train = x_train * 2 - 1.

    with open(TEST_SET, 'rb') as file:
        x_test = np.load(file)

    x_test = x_test [ : batch_size * int(len(x_test) / batch_size) ]
    x_test = x_test * 2 - 1.

    return x_train.astype(np.float32), x_test.astype(np.float32)

def postprocess_data(x):
    return (x + 1.) / 2.

def check_shapes(x_train, batch_size, vae, decoder, latent_dim):
    x_true = x_train[:batch_size]
    print( x_true.shape )
    x_pred = vae.predict(x_true)
    print("MSE", np.mean( np.square( x_true - x_pred ) ) )

    latent_var = np.random.uniform(low=-2., high=2., size=(3, latent_dim))
    print(latent_var.shape)
    x_pred = decoder.predict(latent_var)
    print(x_pred.shape)

def train_vae(vae, x_train, x_test, epochs, batch_size, experiment_number, lr_max):
    callback = KerasSlantedTriangularLearningRateCallback(lr_max = lr_max)
    hist = vae.fit(x=x_train, y=x_train,
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(x_test, x_test),
                   verbose=1,
                   callbacks=[callback])

    #plot_lr(callback.lr_history, experiment_number)
    plot_training_history(hist.history, experiment_number, ylog=True)

def plot_training_history(history, experiment_number, ylog=False):
    figure = plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if ylog:
        plt.yscale('log')
    plt.grid(True)
    #plt.show()

    with open( FOLDER + 'training_history_{}.png'.format( experiment_number ), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
    plt.close(figure)

def plot_lr(lr_history, experiment_number):
    figure = plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(lr_history)
    plt.title('STLR')
    plt.xlabel('# of iterations')
    plt.ylabel('Learning rate')
    plt.grid(True)
    #plt.show()

    with open( FOLDER + 'lr_history_{}.png'.format( experiment_number ), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
    plt.close(figure)

def plot_images(images, experiment_number):
    figure = plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(5, 5, i+1)
        image = images[i, :]
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB) 
        plt.imshow(np.clip(image, 0, 1))
        plt.axis('off')
    #plt.show()

    with open( FOLDER + 'generated_faces_{}.png'.format( experiment_number ), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
    plt.close(figure)

def latent_space(possible_values, dim):
    grid = [ [possible_values[0]] * dim ]

    def produce_latents( latent, i ):
        if i >= len(latent):
            return

        for j in possible_values:
            new_latent = latent[:]
            new_latent[i] = j

        if latent[i] != j:
            grid.append( new_latent )

        produce_latents(new_latent, i+1)

    init_latent = [possible_values[0]] * dim
    produce_latents(init_latent, 0)

    return np.array(grid)

def run_experiment( experiment_number
                    ,latent_dim
                    ,conv_filters
                    ,conv_kernel
                    ,dropout
                    ,batch_size
                    ,epochs
                    ,lr_max
                    ,vae_kl_coef
                    ,BN
                    ,additional_dense_sf
                    ,train=False):
    
    print (experiment_number
            ,latent_dim
            ,conv_filters
            ,conv_kernel
            ,dropout
            ,batch_size
            ,epochs
            ,lr_max
            ,vae_kl_coef
            ,BN
            ,additional_dense_sf)

    with open( FOLDER + 'hyperparams.txt', 'a') as file:
        file.write( str((experiment_number
                        ,latent_dim
                        ,conv_filters
                        ,conv_kernel
                        ,dropout
                        ,batch_size
                        ,epochs
                        ,lr_max
                        ,vae_kl_coef
                        ,BN
                        ,additional_dense_sf)) + '\n')
    
    sess = tf.InteractiveSession()
    K.set_session(sess)

    input_shape=(batch_size, 80, 64, 3)    
    weights_file = FOLDER + "face_generator_{}.h5".format( experiment_number )
    
    vae, decoder = build_vae_model(input_shape, latent_dim, conv_filters, conv_kernel, dropout, vae_kl_coef, BN, additional_dense_sf)
    vae.summary()
    decoder.summary()
    
    # =========================

    x_train, x_test = load_data(batch_size)
    #images = x_train[:25]
    #images = postprocess_data(images)    
    #plot_images(images)
    
    #check_shapes(x_train, batch_size, vae, decoder, latent_dim)    

    # =========================
    
    if train:
        train_vae(vae, x_train, x_test, epochs, batch_size, experiment_number, lr_max)
        # TODO: callback with checkpoint to stop any time!!
        decoder.save_weights( weights_file )        

    # =========================

    decoder.load_weights( weights_file )
    #latent_samples = np.random.uniform(low=-15., high=15., size=(25, latent_dim))
    latent_samples = np.random.normal(loc=0.0, scale=1.0, size=(25, latent_dim))
    #print(latent_samples.shape)
    images = decoder.predict(latent_samples)
    images = postprocess_data(images)
    #print( images[0] )
    plot_images(images, experiment_number)

    sess.close()
    K.clear_session()
    gc.collect()



if __name__ == '__main__':
    
    for experiment_number in range(208, 220):
        latent_dim = np.random.choice( [64, 128, 192] )
        conv_filters_variants = [ [256, 256, 256, 256], [256, 256, 512, 512], [512, 512, 512, 512] ]
        conv_filters = conv_filters_variants[ np.random.choice( len(conv_filters_variants) ) ]
        conv_kernel = 3
        additional_dense_sf = np.random.choice([ 3, 5 ])
        dropout = None
        vae_kl_coef = np.random.choice([ 0.05, 0.1 ])
        BN = True
        lr_max = 0.01
        batch_size = 16
        epochs = 30
    
        run_experiment( experiment_number=experiment_number,
                        latent_dim=latent_dim,
                        conv_filters=conv_filters,
                        conv_kernel=conv_kernel,
                        dropout=dropout,
                        batch_size=batch_size,
                        epochs=epochs,
                        lr_max=lr_max,
                        vae_kl_coef=vae_kl_coef,
                        BN=BN,
                        additional_dense_sf=additional_dense_sf,
                        train=True)









