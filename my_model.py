import pdb
from keras.models import Sequential
from keras.datasets import mnist

print("--------------------")
print("LOADING DATA...")
print("--------------------")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"INPUTS FOR TRAINING: {type(x_train)} {x_train.shape} {x_train.dtype}") #> <class 'numpy.ndarray'> (60000, 28, 28) uint8
print(f"OUTPUTS FOR TRAINING: {type(y_train)} {y_train.shape} {y_train.dtype}") #> <class 'numpy.ndarray'> (60000,) uint8
print(f"INPUTS FOR TESTING: {type(x_test)} {x_test.shape} {x_test.dtype}") #> <class 'numpy.ndarray'> (10000, 28, 28) uint8
print(f"OUTPUTS FOR TESTING: {type(y_test)} {y_test.shape} {y_test.dtype}") #> <class 'numpy.ndarray'> (10000,) uint8

#print("--------------------")
#print("PROCESSING DATA...")
#print("--------------------")

# todo

print("--------------------")
print("CREATING MODEL...")
print("--------------------")

model = Sequential()

print(type(model)) #> <class 'keras.engine.sequential.Sequential'>
