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
TRAIN_SET = '../../experiments/train_lfw.npy'
TEST_SET = '../../experiments/test_lfw.npy'


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
                   verbose=2,
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
    plt.show()

    #with open( FOLDER + 'generated_faces_{}.png'.format( experiment_number ), 'wb') as file:
    #    figure.savefig(file, bbox_inches='tight')
    #plt.close(figure)

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

    input_shape=(batch_size, 64, 64, 3)    
    weights_file = FOLDER + "face_generator_{}.h5".format( experiment_number )
    
    vae, decoder = build_vae_model(input_shape, latent_dim, conv_filters, conv_kernel, dropout, vae_kl_coef, BN, additional_dense_sf)
    #vae.summary()
    #decoder.summary()
    
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

    """
    experiment #1:
    latent_dim = 8
    conv_filters = [64, 128, 256, 512]
    conv_kernel = 5 (single_convs)
    batch_size = 128
    epochs = 2000
    BN

    experiment #2:    
    latent_dim = 64
    conv_filters = [128, 256, 512, 1024]
    conv_kernel = 3 (double_convs)
    dropout = 0.1
    batch_size = 32
    epochs = 500
    BN

    Good?: 9 128 [64, 128, 256] 3 0.1 32 2 0.01 0.01 True

    ??? 13 32 [64, 128, 256] 3 0.1 32 20 0.001 0.01 True

    !!!: 15 256 [64, 128, 256] 3 None 32 20 0.01 0.01 True
        Epoch 20/20 - 22s - loss: 0.0590 - val_loss: 0.0568

    26 128 [64, 128, 256] 3 None 32 20 0.001 0.01 True
        loss: 0.0776 - val_loss: 0.0748
    -------------
    ???
    17 128 [64, 128, 256, 512] 3 None 32 20 0.1 0.1 True

    ------------

    Bad:
    32 8 [64, 128, 256] 3 0.5 32 20 0.001 0.5 True
    30 32 [64, 128, 256] 3 0.1 32 20 0.1 2.0 True
    25 8 [64, 128, 256] 3 None 32 20 0.1 2.0 True
    ? 16 256 [64, 128, 256] 3 0.1 32 20 0.1 0.5 True
    14 8 [64, 128, 256, 512] 3 0.1 32 20 0.001 10.0 True
    ==================================


    Good:
    ?(48, 384, [64, 128, 256], 3, 0.1, 32, 20, 0.01, 0.001, True)
     (55, 256, [64, 128, 256], 3, None, 32, 20, 0.01, 0.01, True)
    ?(59, 256, [32, 64, 256, 512], 3, None, 32, 20, 0.01, 0.01, True)
    ?(62, 384, [32, 64, 128, 256], 3, None, 32, 20, 0.01, 0.001, True)
    ?(64, 256, [64, 128, 256], 3, 0.1, 32, 20, 0.01, 0.001, True)
    ?(65, 384, [32, 64, 128, 256], 3, None, 32, 20, 0.01, 0.1, True)
    ?(66, 512, [64, 128, 256], 3, None, 32, 20, 0.01, 0.001, True) - quickly decreasing
    ?(69, 384, [64, 128, 256], 3, None, 32, 20, 0.01, 0.1, True)
    ?(70, 512, [32, 64, 128, 256], 3, None, 32, 20, 0.01, 0.001, True) - quickly decreasing
    ?(73, 384, [64, 128, 256], 3, 0.1, 32, 20, 0.01, 0.01, True)
    ?(78, 512, [64, 128, 256], 3, None, 32, 20, 0.01, 0.01, True)
    ?(79, 512, [32, 64, 256], 3, None, 32, 20, 0.01, 0.1, True)
    ?(81, 512, [64, 128, 256], 3, None, 32, 20, 0.01, 0.001, True)
    ?(82, 512, [32, 64, 256], 3, None, 32, 20, 0.01, 0.01, True)


    Bad:
    (45, 256, [32, 64, 128, 256], 3, 0.2, 32, 20, 0.01, 0.01, True)
    (46, 128, [32, 64, 256], 3, 0.3, 32, 20, 0.01, 0.001, True)
    (51, 256, [64, 128, 256], 3, 0.2, 32, 20, 0.01, 0.1, True)
    (56, 256, [32, 64, 128, 256], 3, 0.2, 32, 20, 0.01, 0.1, True)
    (68, 512, [64, 128, 256], 3, 0.2, 32, 20, 0.01, 0.01, True)
    (74, 128, [32, 64, 128, 256], 3, 0.3, 32, 20, 0.01, 0.1, True)
    (86, 128, [32, 64, 256, 512], 3, 0.3, 32, 20, 0.01, 0.1, True)
    (91, 512, [32, 64, 128, 256], 3, 0.2, 32, 20, 0.01, 0.01, True)
    (99, 128, [32, 64, 256], 3, 0.2, 32, 20, 0.01, 0.001, True)

    ----------------------------
    
    Good:
    (150, 512, [32, 64, 256, 512], 3, None, 32, 80, 0.01, 0.01, True)
        ~(160, 512, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.01, True)
        ~(163, 256, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.01, True)
    ? (154, 512, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.1, True)
    ? (155, 512, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.1, True)
    (157, 512, [32, 64, 256, 512], 3, 0.1, 32, 80, 0.01, 0.001, True)
        ! only faces! continur to train?
        ~(159, 512, [32, 64, 256, 512], 3, 0.1, 32, 80, 0.01, 0.001, True)
        ~(162, 384, [32, 64, 256, 512], 3, 0.1, 32, 80, 0.01, 0.001, True)
        ?? ~(165, 384, [32, 64, 256, 512], 3, 0.1, 32, 80, 0.01, 0.001, True)
    (158, 256, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.1, True)
        ~(164, 512, [32, 64, 256, 512], 3, None, 32, 80, 0.01, 0.1, True)
    
    Bad:
    (151, 512, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.001, True)
    (152, 512, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.001, True)
    (153, 896, [64, 128, 256], 3, None, 32, 80, 0.01, 0.01, True)
    (156, 384, [64, 128, 256], 3, None, 32, 80, 0.01, 0.001, True)
    (161, 512, [32, 64, 128, 256], 3, None, 32, 80, 0.01, 0.001, True)
    ===============

    (166, 512, [64, 128, 256], 3, 0.1, 32, 80, 0.01, 0.1, True)
    (175, 256, [32, 64, 128, 256, 512], 3, None, 32, 100, 0.01, 0.009, True)


    (183, 128, [32, 64, 256, 512], 3, None, 32, 100, 0.01, 0.001, True)
    

    """

    #for experiment_number in range(200, 250):
    #    latent_dim = np.random.choice( [128, 256] )
    #    conv_filters = np.random.choice( [ [32, 64, 128, 256, 512], [256, 256, 512, 512], [128, 256, 512, 1024], [64, 128, 256, 512, 1024], [32, 64, 128, 256, 512, 1024] ] )
    #    conv_kernel = 3
    #    additional_dense_sf = np.random.choice([ 2, 3, 4, 5 ])
    #    dropout = None # np.random.choice([ None, 0.1, 0.2 ])
    #    vae_kl_coef = np.random.choice([ 0.1, 0.3, 0.5, 0.7 ])
    #    BN = True # np.random.choice([ False, True ])
    #    lr_max = np.random.choice([ 0.003, 0.005, 0.01 ])
    #    batch_size = 32
    #    epochs = 100
    #
    #    run_experiment( experiment_number
    #                    ,latent_dim
    #                    ,conv_filters
    #                    ,conv_kernel
    #                    ,dropout
    #                    ,batch_size
    #                    ,epochs
    #                    ,lr_max
    #                    ,vae_kl_coef
    #                    ,BN
    #                    ,additional_dense_sf)

    run_experiment( experiment_number=200
                    ,latent_dim=128
                    ,conv_filters=[512, 512, 512, 512]
                    ,conv_kernel=3
                    ,dropout=None
                    ,batch_size=32
                    ,epochs=300
                    ,lr_max=0.001
                    ,vae_kl_coef=0.1
                    ,BN=True
                    ,additional_dense_sf=5
                    ,train=True)









