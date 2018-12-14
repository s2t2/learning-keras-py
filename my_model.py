# https://github.com/s2t2/learning-keras-py/blob/master/my_model.py
# https://learningkerasnn-profrossetti.notebooks.azure.com/j/notebooks/my_model.ipynb

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
# view plots in jupyter notebook:
# %matplotlib inline

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
    print("PREPARING/PREPROCESSING DATA...")
    print("--------------------")

    # inputs consists of 60000 image items, each is a 28x28 grid (784)
    # ... which needs to be squashed into a single-dimensional array/layer
    h,w = 28,28
    x_train = x_train.reshape(60000, h * w) #> 6000 entries of 784 size items
    x_test = x_test.reshape(10000, h * w) #> 10000 entries of 784 size items
    verbose_inspect("X (Train)", x_train)
    verbose_inspect("X (Test)", x_test)
    #print(x_train[0]) #> grayscale, values between 0 and 255

    # re-scale data between 0 and 1 (b/c that's how the model wants it)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    #print(x_train[0])

    # need output of our model to go into one of ten bins (digits 0-9)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    verbose_inspect("Y (Train)", y_train)
    verbose_inspect("Y (Test)", y_test)

    print("--------------------")
    print("CREATING MODEL...")
    print("--------------------")

    model = Sequential()
    model.add( Dense(512, activation="relu", input_shape=(784,)) ) # 784 pixel flattened image
    model.add( Dense(512, activation="relu") )
    model.add( Dense(10, activation="softmax") ) # softmax for classification (digits 0-9)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) # categorical_crossentropy (b/c output into 10 categories), other: binary_crossentropy, mse
    model.summary()

    print("--------------------")
    print("TRAINING MODEL...")
    print("--------------------")
    history = model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_test, y_test) )
    print(type(history))

    print("--------------------")
    print("EVALUATING MODEL...")
    print("--------------------")

    if PLOTTING==True:
        plt.plot(history.history["acc"])
        plt.plot(history.history["val_acc"])
        plt.plot(history.history["loss"])

    score = model.evaluate(x_test, y_test)
    print(score)

    # PREDICTIONS: TODO

    print("--------------------")
    print("HAVE A NICE DAY!")
    print("--------------------")

def verbose_inspect(subset_label, subset):
    print(f"{subset_label}: {type(subset)} of {subset.dtype} with shape {subset.shape}")

if __name__ == "__main__":
    main()
