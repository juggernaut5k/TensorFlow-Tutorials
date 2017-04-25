
import os

import numpy as np
import scipy.misc as misc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_rgb_onehot(base_dir):
    base_dir = r'F:\worldview2\Basic, Dallas, ,USA, 40cm_053951940010\ml_datasets\training\grayscale'
    base_dir = r'D:\data\machine_learning\worldview\dallas\grayscale'
    # base_dir = r'C:\dev\data\cnn_deconv'
    # base_dir = r'D:\data\machine_learning\worldview\dallas\rgb_onehot'
    base_dir = '/home/ec2-user/src/data/wv2/rgb_onehot'

    in_dir = base_dir
    true_files = []
    true_targets = []
    false_files = []
    false_targets = []

    # Grab the true
    for dirpath, dirnames, filenames in os.walk(in_dir):
        for filename in [f for f in filenames if f.startswith("true")]:
            true_files.append(os.path.join(dirpath, filename))
            true_targets.append(1)

    # Grab the false data
    for dirpath, dirnames, filenames in os.walk(in_dir):
        for filename in [f for f in filenames
                         if (f.startswith("false"))]:
            false_files.append(os.path.join(dirpath, filename))
            false_targets.append(0)

    # Ensure they're both in the same order
    true_files.sort()
    false_files.sort()

    print(len(true_files))
    print(len(false_files))

    all_files = true_files + false_files
    all_targets = true_targets + false_targets
    shuffle_files, shuffle_targets = shuffle(all_files, all_targets)

    train_files, test_files, train_targets, test_targets = train_test_split(
        shuffle_files, shuffle_targets, train_size=0.90,
        stratify=shuffle_targets)

    img_size = 96

    # Load in the train images
    images_train = np.zeros((len(train_files), img_size, img_size, 3),
                            dtype='uint8')
    for ix, f in enumerate(train_files):
        images_train[ix, :, :, :] = misc.imread(f)

    y = np.array(train_targets).astype('uint8')
    labels_train = np.zeros((len(y), 2), dtype='uint8')
    for ix, y_val in enumerate(y):
        labels_train[ix, y[ix]] = 1

    cls_train = labels_train.argmax(1)

    # Load in the test images
    images_test = np.zeros((len(test_files), img_size, img_size, 3),
                           dtype='uint8')
    for ix, f in enumerate(test_files):
        images_test[ix, :, :, :] = misc.imread(f)

    y = np.array(test_targets).astype('uint8')
    labels_test = np.zeros((len(y), 2), dtype='uint8')
    for ix, y_val in enumerate(y):
        labels_test[ix, y[ix]] = 1

    return images_train, labels_train, images_test, labels_test
