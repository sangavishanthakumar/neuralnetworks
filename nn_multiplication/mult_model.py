''' used the tutorial from https://www.youtube.com/watch?v=8Qc2fG3ZbTg to get an insight in a very simple neural network'''
import tensorflow as tf
from tensorflow import keras

# single layer with one neuron
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = [1, 2, 3]  # input
ys = [2, 4, 6] # correct output for the input after multiplying by 2
#ys = [3, 4, 5]  # correct output for the input after adding 2
model.fit(xs, ys, epochs=1000)

print(model.predict([7]))  # predict the output for the input 7
