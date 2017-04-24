
import glob
import os
import sys

import numpy as np
import scipy.io as io
import scipy.misc as misc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/ucmerced/"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 256

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 21


def load_data(base_dir):
    if os.path.exists('data/ucmerced/data.mat'):
        print('Returning previously generated data.')
        data = io.loadmat('data/ucmerced/data.mat')
        return data['images_train'], data['images_test'], \
               data['classes_train'], data['classes_test']
    print('Generating new data.')
    class_dirs = glob.glob(os.path.join(base_dir, '*'))
    class_dirs.sort()

    images = []
    classes = []
    for ix, class_dir in enumerate(class_dirs):
        img_files = glob.glob(os.path.join(class_dir, '*.tif'))
        for iy, img_file in enumerate(img_files):
            img = misc.imread(img_file)
            if img.shape[0] != 256 | img.shape[1] != 256:
                img = misc.imresize(img, (256, 256))
            images.append(img)
            one_hot = np.zeros(len(class_dirs), dtype='uint8')
            one_hot[ix] = 1
            classes.append(one_hot)

    shuffle_images, shuffle_classes = shuffle(images, classes)

    images_train, images_test, classes_train, classes_test = train_test_split(
        shuffle_images, shuffle_classes, train_size=0.90,
        stratify=shuffle_classes)

    images_train = np.array(images_train)
    images_train = images_train.squeeze()
    images_test = np.array(images_test).squeeze()
    classes_train = np.array(classes_train).squeeze()
    classes_test = np.array(classes_test).squeeze()

    io.savemat('data/ucmerced/data.mat', {'images_train': images_train,
                                          'images_test': images_test,
                                          'classes_train': classes_train,
                                          'classes_test': classes_test})

    return images_train, images_test, classes_train, classes_test


def load_class_names(base_dir):
    class_dirs = glob.glob(os.path.join(base_dir, '*'))
    class_dirs.sort()

    class_names = [os.path.basename(class_dir) for class_dir in class_dirs]
    return class_names
