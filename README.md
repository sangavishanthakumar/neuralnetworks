# Neural Networks

This project contains some small models to get an understanding about neural networks. It currently contains three models: mnist_model.py, add_model.py, mult_model.py.

## mult_model.py
This model uses keras and tensorflow. It is based on the tutorial https://www.youtube.com/watch?v=8Qc2fG3ZbTg and learns multiplication with a small dataset (three integers). 
It contains a single layer with one neuron. 

SGD is used as the optimizer and MSE as the loss function.

## mnist_model.py
This model recognizes handwritten digits between 0 to 9 and is based on this tutorial: https://www.youtube.com/watch?v=w8yWXqWQYmU 
The used dataset is from MNIST database and can be accessed here: https://www.kaggle.com/c/digit-recognizer/data?select=train.csv

This model uses only numpy. It consists of two layers. In the forward propagation, the first layer uses ReLU $\max(0, Z)$ as the activation function. In the 
second layer, softmax $\frac{e^Z}{\sum(e^Z)}$ is used. To update $w_1, b_1, w_2$ and $b_2$, gradient descent $x_{t+1} = x_t - \eta \nabla f(x_t)$ is used.

At the moment, I am implementing this with PyTorch Lightning.

## add_model.py
This model should add up two numbers. It is currently in progress.
