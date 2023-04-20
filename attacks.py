# !pip install cleverhans
# Tensorflow 2.12

import os

# from google.colab import drive
# drive.mount('/content/drive')
import keras.backend
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

import glob
# import imageio
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import time

class SubNet(Model):
    def __init__(self):
        super(SubNet, self).__init__()
        self.conv1 = Conv2D(64, 8, strides=(2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(128, 6, strides=(2, 2), activation="relu", padding="valid")
        self.conv3 = Conv2D(128, 5, strides=(1, 1), activation="relu", padding="valid")
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.5)
        self.flatten = Flatten()
        self.dense = Dense(10)

    def call(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        return self.dense(x)


def ld_mnist(batch=150):
    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset, info = tfds.load("mnist", with_info=True, as_supervised=True)
    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(batch)
    mnist_test_all = mnist_test.map(convert_types).shuffle(10000, reshuffle_each_iteration=False)
    mnist_test = mnist_test_all.take(9850)
    mnist_train1 = mnist_test_all.skip(9850)
    mnist_test = mnist_test.batch(batch)
    mnist_train1 = mnist_train1.batch(batch)
    return EasyDict(train=mnist_train, test=mnist_test, train1=mnist_train1)


def main():
    data = ld_mnist(1)
    mod = SubNet()
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.metrics.Mean(name="train_loss")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = mod(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, mod.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mod.trainable_variables))
        train_loss(loss)

    for epoch in range(100):
        progress_bar_train = tf.keras.utils.Progbar(150)
        for (x, y) in data.train1:
            train_step(x, y)
            progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])

    progress_bar_test = tf.keras.utils.Progbar(9850)
    perturbed = []
    original = []
    y_list = []
    for x, y in data.test:
        x_fgm = fast_gradient_method(mod, x, 0.3, np.inf)
        # plt.imshow(x[0, :, :, 0], cmap='gray', vmin=-0.3, vmax=1.3)
        # plt.show()
        # plt.imshow(x_fgm[0, :, :, 0], cmap='gray', vmin=-0.3, vmax=1.3)
        # plt.show()
        # myloss = tf.keras.losses.MeanSquaredError()
        # loss = myloss(x_fgm, x)
        perturbed.append(x_fgm)
        original.append(x)
        y_list.append(y)
        progress_bar_test.add(x.shape[0])
    perturbed = np.array(perturbed)
    original = np.array(original)
    y_list = np.array(y_list)
    with open('perturbed.npy', 'wb') as f:
        np.save(f, perturbed)
    with open('original.npy', 'wb') as f:
        np.save(f, original)
    with open('y_list.npy', 'wb') as f:
        np.save(f, y_list)

    return x
    
x = main()

