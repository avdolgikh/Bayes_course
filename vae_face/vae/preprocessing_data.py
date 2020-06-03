import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show(img):
    plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB) ) # , cmap='gray'    plt.show()

def imread(path):
    return cv2.imread( path ).astype(np.float32)  / 255.

def resize(img):
    return cv2.resize(img, (64, 64))

def read_paths(folder):
    paths = []
    for subfolder in os.listdir(folder):
        subfolder_path = folder + "/" + subfolder
        if os.path.isfile(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
            file_path = subfolder_path + "/" + filename
            if os.path.isfile(file_path):
                paths.append( file_path )
    return paths

def read_images(folder):
    images = []
    for subfolder in os.listdir(folder):
        subfolder_path = folder + "/" + subfolder
        if os.path.isfile(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
            file_path = subfolder_path + "/" + filename
            if os.path.isfile(file_path):                
                img = imread( file_path )
                images.append( resize(img) )
    return np.array(images)

def preprocess():
    folder = '../../lfw'
    images = read_images(folder)
    # TODO: or take only a single image from a folder?

    print(images.shape)

    # split on train and test (20%)
    np.random.shuffle(images)
    train_size = int(images.shape[0] * .8)
    train = images[ : train_size ]
    test = images[ train_size : ]
    print(train.shape)
    print(test.shape)

    # save as np-files
    with open('../../lfw/train_lfw.npy', 'wb') as file:
        np.save(file, train)

    with open('../../lfw/test_lfw.npy', 'wb') as file:
        np.save(file, test)



if __name__ == '__main__':

    with open('../../lfw/test_lfw.npy', 'rb') as file:
        test = np.load(file)

    for _ in range(10):
        index = np.random.choice(len(test))        
        show(test[index])

    # build VAE

    # research latent space (t-sne?)
    # what is a search in the latent space? Gradient search, kNN?


