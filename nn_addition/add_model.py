import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random

''' IN PROGRESS '''

''' creating a neural network that learns how to add two numbers by using PyTorch Lightning '''

''' create a train and test dataset to train the network '''


class AdditionDataset(Dataset):
    ''' let the neuronal network train with numbers that are between 0 and 100'''

    def __init__(self, num_examples, max_num=100):
        self.num_examples = num_examples
        self.data = torch.randint(0, max_num, (num_examples, 2), dtype=torch.float32)
        # torch create a tensor
        # randint creates random integers
        # randint(0, max_num, (num_examples, 2), dtype=torch.float32)
        # # 0: min value is 0
        # # max_num: max value is max_num-1
        # # (num_samples, 2): create a tensor with num_samples rows and 2 columns e.g. num_samples= 3
        # # #-> [[1,2],[2,3],[4,5]] in the first brackets 3 elements exist (dim[0]=3) and in each elements 2 numbers exist (dim[1]=2)
        # # dtype=torch.float32: datatype of tensor is float because neuronal networks work better with floats
        self.labels = self.data.sum(dim=1)
        # sum the "inner" elements: [[1,2],[2,3],[4,5]] -> [3,5,9]

    ''' function to get the element on the index "index" in data (samples, e.g. [1,2]) and labels (sum, e.g. 3)'''

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    ''' return amount of samples that are in the dataset '''

    def __len__(self):
        return self.num_examples
        # returning self.data would return the tensor with the data and this would not give the amount of samples


'''LightningModule is an extension of PyTorch that has the training, validation, testing and the prediction in one 
class'''


class AdditionModel(pl.LightningModule):
    # constructor of class AdditionModel
    # input: 2 (the 2 numbers that should be added)
    # hidden layers: 10 neurons
    # output: 1 (the sum of the 2 numbers)
    def __init__(self, input_=2, hidden_layers=10, output=1):
        # call constructor of pl.LightningModule to inherit properties of LightningModule
        super(AdditionModel, self).__init__()

        # define the neural network
        self.network = nn.Sequential(nn.Linear(input_, hidden_layers),
                                     nn.ReLU(),
                                     nn.Linear(hidden_layers, output))

        # nn.Linear: performs a linear transformation on input_ and maps input1 on every hidden_layers and input2 on every hidden_layer
        # linear transformation: y = xA^T +b (x: input vector, A: weight matrix, b: bias vector, y: output vector)
        # nn.ReLU: activation function that returns for negative numbers 0 and for positive numbers the number itself
        # second linear transformation: maps hidden layers on output (10 neurons -> 1 neuron)

    # define the forward function
    def forward(self, x):
        # ! nn.Linear returns the batch size and the output size
        # nn.Linear(2,1) -> makes [[1,2]] to [[3]] (batch size = 1, output size = 1).
        # nn.Linear(2,1) -> makes [[1,2],[2,3]] to [[3],[5]] (batch size = 2, output size = 1).
        # remove the output size by using .squeeze(1) that removes the output size dimension so the example from above would be [2] instead of [2,1] bzw. it would be [3,5] instead of [[3],[5]]
        return self.network(x).squeeze(1)

    def training_step(self, batch, batch_index):
        # batch: tuple that contains the input data and label -> batch = (input, label)
        # batch in general: multiple samples that are processed at the same time
        # batch_index is the index of the batch

        # extract the input and label from the batch
        x, y = batch

        # call the forward function to get the prediction
        y_pred = self(x)  # self calls implicitly the forward function

        # calculate the loss with the mean squared error
        loss = nn.MSELoss()(y_pred, y)

        return loss

    def configure_optimizers(self):
        # use optimizer to optimize the network and the output
        return optim.Adam(self.parameters(), lr=0.001)
