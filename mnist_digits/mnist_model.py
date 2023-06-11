"""
tutorial: https://www.youtube.com/watch?v=w8yWXqWQYmU
dataset: https://www.kaggle.com/c/digit-recognizer/data?select=train.csv
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

''' The train.csv file is not in this repo. To run this code, download the dataset from the link above '''
data = pd.read_csv('digit-recognizer/train.csv')
# print(data.head())

data = np.array(data)
m, n = data.shape  # get the dimensions; m = rows; n = amount of features + 1
np.random.shuffle(data)

# create data dev
data_dev = data[0:1000].T  # transpose the data
# print("Transposed data: ", data_dev)
# print("Shape of data_dev: ", data_dev.shape)
Y_dev = data_dev[0]  # first row 5 3 6 ... 6 1 9
X_dev = data_dev[1:n]  # pixel p0 ... pn
X_dev = X_dev / 255.  # normalize the data

# create data train
data_train = data[1000:m].T  # transpose the data
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.  # normalize the data


# print(X_train[:,0].shape) # 784 rows

def init_param():
    # interval [-0.5, 0.5]
    W1 = np.random.rand(10, 784) - 0.5  # weight1 for 10 rows for 10 digits, 784 columns for 784 = 28*28 pixels
    b1 = np.random.rand(10,
                        1) - 0.5  # bias2 for the next layer so each pixel is applied to a neuron from 0 to 9 (next
    # layer has 10 neurons)
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)  # checking element-wise the maximum of array elements


def softmax(Z):
    # softmax function: e^Z/sum(e^Z)
    return np.exp(Z) / sum(np.exp(Z))


def forward_prop(W1, b1, W2, b2, X):
    # steps: compute new output by using weights, biases and the activation function
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)  # first activation function
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)  # second activation function
    return Z1, A1, Z2, A2


# https://www.educative.io/blog/one-hot-encoding
# one hot: make all columns 0 except the one that represents the label
# example: 5 -> 0 0 0 0 0 1 0 0 0 0
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1  # depending on the label Y set the value on 1
    one_hot_Y = one_hot_Y.T  # flip it so each column is an example
    return one_hot_Y


# implement the derivative of the ReLU function
def derivative_ReLU(Z):
    return Z > 0


# code backpropagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# define gradient descent
def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_param()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
