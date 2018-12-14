# https://github.com/s2t2/learning-keras-py/blob/master/my_model.py

import pdb
import os

from keras.datasets import mnist  #> "Importing Tensorflow Backend"
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

import matplotlib
matplotlib.use('TkAgg') # bypasses ImportError: Python is not installed as a framework....
import matplotlib.pyplot as plt

PLOTTING = (os.environ.get("PLOTTING") == "true") or False

def main():
    print("--------------------")
    print("LOADING DATA...")
    print("--------------------")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    verbose_inspect("X (Train)", x_train)
    verbose_inspect("X (Test)", x_test)
    print("---")
    verbose_inspect("Y (Train)", y_train)
    verbose_inspect("Y (Test)", y_test)

    if PLOTTING == True:
        imageset = x_train
        plt.subplot(141)
        plt.imshow(imageset[0], cmap="gray")
        plt.subplot(142)
        plt.imshow(imageset[1], cmap=plt.get_cmap('gray'))
        plt.subplot(143)
        plt.imshow(imageset[2], cmap=plt.get_cmap('gray'))
        plt.subplot(144)
        plt.imshow(imageset[3], cmap=plt.get_cmap('gray'))
        plt.show()

    print("--------------------")
    print("PREPARING DATA...")
    print("--------------------")
    # inputs consists of 60000 image items, each is a 28x28 grid
    # ... which needs to be squashed into a single-dimensional array/layer
    h,w = 28,28
    x_train = x_train.reshape(60000, h*w)
    x_test = x_test.reshape(10000, h*w)
    verbose_inspect("X (Train)", x_train)
    verbose_inspect("X (Test)", x_test)

    print("--------------------")
    print("CREATING MODEL...")
    print("--------------------")

    model = Sequential()
    model.add( Dense(512, activation="relu", input_shape=(784,)) ) # 784 pixel flattened image
    model.add( Dense(512, activation="relu") )
    model.add( Dense(10, activation="softmax") ) # softmax for classification (digits 0-9)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) # categorical_crossentropy (b/c output into 10 categories), other: binary_crossentropy, mse
    model.summary()

    #
    # TRAIN MODEL
    # ... takes 20 mins per epoch (or skip to loading the weights :-D)

    print("--------------------")
    print("TRAINING MODEL...")
    print("--------------------")
    history = model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_test, y_test) )
    print(type(history))

    #
    # EVALUATE MODEL
    #

    print("--------------------")
    print("EVALUATING MODEL...")
    print("--------------------")

    #if PLOTTING==True:
    #    plt.plot(history.history["acc"])
    #    plt.plot(history.history["val_acc"])
    #    plt.plot(history.history["loss"])

    score = model.evaluate(x_test, y_test)
    print(score)

    #
    # PREDICT
    #

    # todo

def verbose_inspect(subset_label, subset):
    print(f"{subset_label}: {type(subset)} of {subset.dtype} with shape {subset.shape}")

if __name__ == "__main__":
    main()
