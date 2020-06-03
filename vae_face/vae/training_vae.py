import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (13, 5)

from building_vae_graphs import build_vae_model

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

def train_vae(vae, x_train, x_test, epochs, batch_size, experiment_number):
    hist = vae.fit(x=x_train, y=x_train,
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(x_test, x_test),
                   verbose=1)
    
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

def plot_images(images):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(5, 5, i+1)
        image = images[i, :]
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB) 
        plt.imshow(np.clip(image, 0, 1))
        plt.axis('off')
    plt.show()

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

    """
    experiment_number = 3
    
    latent_dim = 32
    conv_filters = [32, 64, 128, 256]
    conv_kernel = 3
    dropout = None
    batch_size = 32
    epochs = 3

    input_shape=(batch_size, 64, 64, 3)    
    weights_file = FOLDER + "face_generator_{}.h5".format( experiment_number )
    
    vae, decoder = build_vae_model(input_shape, latent_dim, conv_filters, conv_kernel, dropout)
    vae.summary()
    decoder.summary()
    
    # =========================

    x_train, x_test = load_data(batch_size)
    #images = x_train[:25]
    #images = postprocess_data(images)    
    #plot_images(images)
    
    #check_shapes(x_train, batch_size, vae, decoder, latent_dim)    

    # =========================
    
    train_vae(vae, x_train, x_test, epochs, batch_size, experiment_number)
    #decoder.save_weights( weights_file )

    # TODO: callback with checkpoint to stop any time!!

    # =========================

    decoder.load_weights( weights_file )
    latent_samples = np.random.uniform(low=-15., high=15., size=(25, latent_dim))
    #latent_samples = np.random.normal(loc=0.0, scale=1.0, size=(25, latent_dim))
    print(latent_samples.shape)
    images = decoder.predict(latent_samples)    
    images = postprocess_data(images)
    print( images[0] )
    plot_images(images)







