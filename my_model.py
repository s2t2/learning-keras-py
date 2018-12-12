import pdb
import os
from keras.models import Sequential
from keras.datasets import mnist

PLOTTING = (os.environ.get("PLOTTING") == "true") or False

print("--------------------")
print("LOADING DATA...")
print("--------------------")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def verbose_inspect(subset_label, subset):
    print(f"{subset_label}: {type(subset)} of {subset.dtype} with shape {subset.shape}")

verbose_inspect("X (Train)", x_train)
verbose_inspect("Y (Train)", y_train)
verbose_inspect("X (Test)", x_test)
verbose_inspect("Y (Test)", y_test)

if PLOTTING == True:
    import matplotlib
    matplotlib.use('TkAgg') # bypasses ImportError: Python is not installed as a framework....
    import matplotlib.pyplot as plt
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
print("CREATING MODEL...")
print("--------------------")

model = Sequential()
print(type(model)) #> <class 'keras.engine.sequential.Sequential'>

#model.add(Dense(3, input_dim=2, activation="relu"))
#model.add(Dense(3, activation="relu"))
#model.add(Dense(1)) # activation="softmax" for classification

# model.compile(optimizer="adam", loss="mse") # categorical_crossentropy or binary_crossentropy
