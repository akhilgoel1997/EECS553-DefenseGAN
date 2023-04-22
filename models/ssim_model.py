# -*- coding: utf-8 -*-
"""FGSM_ATTACKS_123.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HT-XthiDZa0E6LU5kVuecVhmGO-Yk5xu
"""


import os

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

import glob
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import time

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')



checkpoint_dir = '/content/drive/My Drive/Colab Notebooks/EECS553_Project/training_checkpoints_new'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

R = 15
L = 500
tf.random.set_seed(112233)

def argminZ(image):
    mainList = []
    SSList=[]
    ZList = []
    for i in range(R):
        myloss = tf.keras.losses.MeanSquaredError()
        myoptim = tf.keras.optimizers.Adam(1e-3, amsgrad = True)
        Z = tf.Variable(tf.random.normal([image.shape[0], 100], seed=112233), trainable=True)
        with tf.GradientTape() as Z_tape:
            generated_image = generator(Z, training=False)
            LOSS = myloss(generated_image, image)
            # LOSS = -1*tf.image.ssim(generated_image, image, max_val = 2.0, filter_size=11, filter_sigma=5)
        gradients = Z_tape.gradient(LOSS, [Z])
        # for j in range(L):
        j=0
        while(tf.image.ssim(generator(Z, training=False), image, max_val = 2.0) < 0.4 and j<=L):
            myoptim.apply_gradients(zip(gradients, [Z]))
            j+=1
        # mainList.append(-1*tf.image.ssim(generator(Z, training=False), image, max_val = 2.0, filter_size=11, filter_sigma=5))
        mainList.append(myloss(generator(Z, training=False), image))
        ZList.append(Z)
    mainList = np.array(mainList)
    for e,m in enumerate(mainList):
      SSList.append(tf.image.ssim(generator(ZList[e], training=False), image, max_val = 2.0))
    
    SSList = np.array(SSList)
    print(SSList)
    final_z = ZList[np.argmax(SSList)]
    return final_z

Xgen = None


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(64, 5, strides=(1, 1), activation="relu", padding="same")
        self.conv2 = Conv2D(64, 5, strides=(2, 2), activation="relu", padding="valid")
        # self.conv3 = Conv2D(128, 5, strides=(1, 1), activation="relu", padding="valid")
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        return self.dense2(x)


def ld_mnist(batch=150):
    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        image = image*2
        image = image-1
        return image, label

    dataset, info = tfds.load(
        "mnist", data_dir="gs://tfds-data/datasets", with_info=True, as_supervised=True
    )
    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(batch)
    mnist_test = mnist_test.map(convert_types).batch(batch)
    return EasyDict(train=mnist_train, test=mnist_test)


def main():
    global Xgen
    data = ld_mnist()
    model = Net()
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.metrics.Mean(name="train_loss")
    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
    test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()
    test_acc_defense_gan = tf.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    for epoch in range(8):
        progress_bar_train = tf.keras.utils.Progbar(60000)
        for (x, y) in data.train:
            train_step(x, y)
            progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])

    # data = ld_mnist(1)
    with open('/content/drive/My Drive/Colab Notebooks/EECS553_Project/perturbed_cwl2.npy', 'rb') as f:
        pert = np.load(f)
    with open('/content/drive/My Drive/Colab Notebooks/EECS553_Project/original_cwl2.npy', 'rb') as f:
        orig = np.load(f)
    with open('/content/drive/My Drive/Colab Notebooks/EECS553_Project/y_list_cwl2.npy', 'rb') as f:
        y_list = np.load(f)
    progress_bar_test = tf.keras.utils.Progbar(10000)
    i = 0
    for x, x_fgm, y in zip(orig, pert, y_list):
        i += 1
        y_pred = model(x)
        test_acc_clean(y, y_pred)
        plt.imshow(x[0, :, :, 0], cmap='gray')
        plt.show()
        # x_fgm = fast_gradient_method(model, x, 0.3, np.inf)

        #x_fgm = carlini_wagner_l2(model, x, clip_min = -1, clip_max = +1)
        plt.imshow(x_fgm[0, :, :, 0], cmap='gray')
        plt.show()
        y_pred_fgm = model(x_fgm)
        test_acc_fgsm(y, y_pred_fgm)
        # myloss= tf.keras.losses.MeanSquaredError()
        # loss = myloss(x_fgm, x)
        # print(loss)
        Z = argminZ(x_fgm)
        Xgen = generator(Z, training=False)
        plt.imshow(Xgen[0, :, :, 0], cmap='gray')
        plt.show()
        # myloss = tf.keras.losses.MeanSquaredError()
        # loss = myloss(x, Xgen)
        # print(loss)
        y_pred_defense_gan = model(Xgen)
        test_acc_defense_gan(y, y_pred_defense_gan)
        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        print("test acc on FGM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
        print("test acc on Defense GAN examples (%): {:.3f}".format(test_acc_defense_gan.result() * 100))

        progress_bar_test.add(x.shape[0])
        if i == 10:
            break
    print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
    print("test acc on FGM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
    print("test acc on Defense GAN examples (%): {:.3f}".format(test_acc_defense_gan.result() * 100))
    return x
    
x = main()



tf.image.ssim(generated_image, image)

cntr = 1
for xx,yy in ld_mnist().test:
  Z=argminZ(xx)
  Xgen = generator(Z, training=False)
    # Z = tf.random.normal([10, 100])
    # generated_image = generator(Z, training=False)
    # print(x.shape)
    # print(generated_image.shape)
    # print("\n")
  print(cntr)
  cntr+=1
  break

Xgen = generator(Z, training=False)
Xgen.shape

xx.shape[0]

Z.shape

plt.imshow(Xgen[0, :, :, 0]+1, cmap='gray')

plt.imshow(x[0, :, :, 0]+1, cmap='gray')

Xgen[4, :, :, 0]+1

x[4, :, :, 0]

test_acc_defense_gan.result() * 100




plt.imshow(nns[0], cmap='gray')

Xgen[0, :, :, :].shape


