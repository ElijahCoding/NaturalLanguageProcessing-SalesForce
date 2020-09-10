import struct
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import expit

def load_data():
    with open('dataset/train-labels-idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('dataset/train-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    with open('dataset/t10k-labels-idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('dataset/t10k-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    return train_images, train_labels, test_images, test_labels

def visualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(64):
        img = img_array[label_array == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()

def encode_one_hot(y, num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    for i, val in enumerate(y):
        one_hot[val, i] = 1.0
    return one_hot

def sigmoid(z):
    # return(1 / (1 + np.exp(-z)))
    return expit(z)

def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)

def visualize_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()

visualize_sigmoid()
# train_x, train_y, test_x, test_y = load_data()
#
# visualize_data(train_x, train_y)