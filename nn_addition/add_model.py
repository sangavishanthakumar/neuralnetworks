import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random
import torch.nn.functional as F

''' IN PROGRESS '''

''' creating a neural network that learns how to add two numbers by using PyTorch Lightning '''

''' create a train and test dataset to train the network '''


class AdditionDataset(Dataset):
    ''' let the neuronal network train with numbers that are between 0 and 100'''

    # TODO max_num erhöhen
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


# TODO insert further metrics
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
        # 1: remove the output size by using .squeeze(1) that removes the output size dimension so the example from above would be [2] instead of [2,1] bzw. it would be [3,5] instead of [[3],[5]]
        # squeeze(1): AVERAGE LOSS low 5.418554827052446e-06
        # squeeze():  AVERAGE LOSS low 0.0129
        # return self.network(x).squeeze(1) # not squeeze() because we work with batches; in flask it is squeeze()
        # squeeze(1) is used in test.py
        # 3:
        return self.network(x)

    # backward? Implementation not needed, because the trainer does this automatically by
    # abstracting the loss valu from training_step and calling .backward() on iit
    def training_step(self, batch, batch_index):
        # batch: tuple that contains the input data and label -> batch = (input, label)
        # batch in general: multiple samples that are processed at the same time
        # batch_index is the index of the batch

        # extract the input and label from the batch
        x, y = batch

        # call the forward function to get the prediction
        y_pred = self(x)  # self calls implicitly the forward function

        # Print shapes for debugging
        # print("y_pred shape:", y_pred.shape)
        y_pred_squeezed = y_pred.squeeze(1)
        # print("y_pred_squeezed shape:", y_pred_squeezed.shape)
        # print("y shape:", y.shape)
        # calculate the loss with the mean squared error
        loss = F.mse_loss(y_pred_squeezed, y)
        # Why not loss = nn.MSELoss()(y_pred, y)?
        # -> nn.MSELoss() is creating an instance, but I do not need the loss value multiple times,
        # so F.mse_loss is enough
        self.log('train_loss', loss)  # log the loss to display it later with tensorboard
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
       #  y = y.unsqueeze(1)  # add a dimension to y to avoid the "axes" error
        y_pred = self(x)
        y_pred_squeezed = y_pred.squeeze(1)
        loss = F.mse_loss(y_pred_squeezed, y)
        self.log('val_loss', loss)  # to display the val_loss, val_data and the dataloader are necessary -> train.py
        return loss

    def configure_optimizers(self):
        # use optimizer to optimize the network and the output
        return optim.Adam(self.parameters(), lr=0.001)  # lr -> learning rate == hyper parameter
