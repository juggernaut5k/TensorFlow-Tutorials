
import glob
import os
import sys

import numpy as np
import scipy.misc as misc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_path = 'data/UC_Merced'


def load(base_dir):
    class_dirs = glob.glob(os.path.join(base_dir, '*'))
    class_dirs.sort()

    class_labels = [os.path.basename(class_dir) for class_dir in class_dirs]

    labels = []
    images = []
    for ix, class_dir in enumerate(class_dirs):
        print('Loading: ', class_dir)
        print('\n')
        img_files = glob.glob(os.path.join(base_dir, class_dir, '*.tif'))
        for iy, img_file in enumerate(img_files):
            sys.stdout.write("\r%d of %d" % (iy + 1, len(img_files)))
            sys.stdout.flush()
            image = misc.imread(img_file)
            image = misc.imresize(image, (256, 256), interp='nearest')
            images.append(image)
            one_hot_label = np.zeros(len(class_dirs), dtype='uint8')
            one_hot_label[ix] = 1
            labels.append(one_hot_label)

    shuffle_images, shuffle_labels = shuffle(images, labels)

    images_train, images_test, labels_train, labels_test = train_test_split(
        shuffle_images, shuffle_labels, train_size=0.90,
        stratify=shuffle_labels)

    return images_train, images_test, labels_train, labels_test, class_labels


def load_class_names(base_dir):
    class_dirs = glob.glob(os.path.join(base_dir, '*'))
    class_dirs.sort()

    class_labels = [os.path.basename(class_dir) for class_dir in class_dirs]
    return class_labels
