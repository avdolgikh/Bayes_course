import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show(img):
    plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB) ) # , cmap='gray'
    plt.show()

def imread(path):
    return cv2.imread( path ).astype(np.float32)  / 255.

def resize(img, size):
    return cv2.resize(img, size)

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

def read_images(folder, flat_structure, size=None):
    images = []
    for subitem in os.listdir(folder):
        subitem_path = folder + "/" + subitem
        if flat_structure:
            if os.path.isfile(subitem_path):                                
                img = imread( subitem_path )
                if size is not None:
                    img = resize( img, size )
                images.append( img )
        else:
            if os.path.isfile(subitem_path):
                continue
            for filename in os.listdir(subitem_path):
                file_path = subitem_path + "/" + filename
                if os.path.isfile(file_path):                
                    img = imread( file_path )
                    if size is not None:
                        img = resize( img, size )
                    images.append( img )
    return np.array(images)

def preprocess(folder_with_originals, set_name, flat_structure, size):    
    images = read_images(folder_with_originals, flat_structure, size)
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
    with open('../../experiments/train_{}.npy'.format(set_name), 'wb') as file:
        np.save(file, train)

    with open('../../experiments/test_{}.npy'.format(set_name), 'wb') as file:
        np.save(file, test)

def resize_batch(file_path, size):
    with open(file_path, 'rb') as file:
        data = np.load(file)
        images = []
        for img in data:
            images.append( resize(img, size) )
        images = np.array(images)
        print(images.shape)
        with open(file_path, 'wb') as file:
            np.save(file, images)




if __name__ == '__main__':

    #preprocess('../../img_align_celeba', "celeba", flat_structure=True, size=(64,78))

    resize_batch('../../experiments/train_celeba.npy', (64, 80))

    #with open('../../experiments/test_celeba.npy', 'rb') as file:
    #    test = np.load(file)

    # for _ in range(10):
    #     index = np.random.choice(len(test))
    #     img = test[index]
    #     show(test[index])

    # build VAE

    # research latent space (t-sne?)
    # what is a search in the latent space? Gradient search, kNN?


