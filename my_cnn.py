import pdb
import os

from keras.models import Sequential
from keras.datasets import mnist #> "Importing Tensorflow Backend"

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
    print(type(model)) #> <class 'keras.engine.sequential.Sequential'>

    #model.add(Dense(3, input_dim=2, activation="relu"))
    #model.add(Dense(3, activation="relu"))
    #model.add(Dense(1)) # activation="softmax" for classification
    #model.compile(optimizer="adam", loss="mse") # categorical_crossentropy or binary_crossentropy

    # CNN
    model.add( Conv2D(32, kernel_size=(5,5), input_shape=(28,28,1), padding="same", activation="relu") ) # INPUT LAYER
    model.add( MaxPooling2D() )
    model.add( Conv2D(64, kernel_size=(5,5), padding="same", activation="relu") )
    model.add( MaxPooling2D() )
    model.add( Flatten() )
    model.add( Dense(1024, activation="relu") )
    model.add( Dense(10, activation="softmax") ) # OUTPUT LAYER softmax for categorical
    model.compile()
    print(model.summary())

    #
    # TRAIN MODEL
    # ... takes 20 mins per epoch (or skip to loading the weights :-D)

    # history_model = model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_train, y_train) )
    # plt.plot(histor_model.history_model["acc"])
    # plt.plot(histor_model.history_model["val_acc"])

    # LOAD WEIGHTS

    model.load_weights("weights/cnn-model5.h5")

    #
    # PREDICT
    #

    score = model.evaluate(x_test, y_test)
    print(score)











def verbose_inspect(subset_label, subset):
    print(f"{subset_label}: {type(subset)} of {subset.dtype} with shape {subset.shape}")

if __name__ == "__main__":
    main()
