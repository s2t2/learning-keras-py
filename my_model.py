import pdb
from keras.models import Sequential
from keras.datasets import mnist

# from matplotlib import pyplot as plt #> ImportError: Python is not installed as a framework....
# h/t: https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python#comment56913201_21789908
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("--------------------")
print("LOADING DATA...")
print("--------------------")

# h/t: https://keras.io/datasets/#mnist-database-of-handwritten-digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"  + TRAINING/INPUTS: {type(x_train)} {x_train.shape} {x_train.dtype}") #> <class 'numpy.ndarray'> (60000, 28, 28) uint8
print(f"  + TRAINING/OUTPUTS: {type(y_train)} {y_train.shape} {y_train.dtype}") #> <class 'numpy.ndarray'> (60000,) uint8
print(f"  + TESTING/INPUTS: {type(x_test)} {x_test.shape} {x_test.dtype}") #> <class 'numpy.ndarray'> (10000, 28, 28) uint8
print(f"  + TESTING/OUTPUTS: {type(y_test)} {y_test.shape} {y_test.dtype}") #> <class 'numpy.ndarray'> (10000,) uint8

print("--------------------")
print("EXAMPLE TRAINING DATA INAGES...")
print("--------------------")

# h/t: https://topicfly.io/classify-mnist-digits-keras/
plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))

plt.show() # h/t: https://stackoverflow.com/a/35085705/670433

#pdb.set_trace()

print("--------------------")
print("PROCESSING DATA...")
print("--------------------")

# todo

print("--------------------")
print("CREATING MODEL...")
print("--------------------")

model = Sequential()

print(type(model)) #> <class 'keras.engine.sequential.Sequential'>
