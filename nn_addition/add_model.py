import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random

''' IN PROGRESS '''

''' creating a neural network that learns how to add two numbers by using PyTorch Lightning '''


class AdditionDataset(Dataset):
    def __init__(self, num_examples):
        self.data = self.generate_data(num_examples)

    def __getitem__(self, index):
        x, y = self.data[index]
        return torch.Tensor([x, y]), torch.Tensor([x + y])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def generate_data(num_examples):
        data = []
        for _ in range(num_examples):
            x = random.randint(0, 100)
            y = random.randint(0, 100)
            print("\nPrint to check if x and y are NaN: ", x, y)
            data.append((x, y))
            print("\nPrint to check if the data is ok: ", data)
        return data
    # TODO check why the tensor content is NaN during the feed forward process


class AdditionModel(pl.LightningModule):
    def __init__(self):
        super(AdditionModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    # two layers
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.MSELoss()(output, y)
        self.log('train_loss', loss)
        print("Loss:", loss.item())  # Print loss
        print("Predicted Output:", output)  # Print predicted output
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

