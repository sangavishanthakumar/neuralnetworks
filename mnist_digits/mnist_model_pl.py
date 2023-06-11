import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

''' IN PROGRESS '''

''' converting mnist_model to pytorch_lightning model'''
''' used dataset: https://www.kaggle.com/c/digit-recognizer/data?select=train.csv'''


# Define the neural network module
class MNISTNet(pl.LightningModule):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        # print prediction
        print("Loss:", loss.item())  # Print loss
        print("Predicted Output:", y_hat)  # Print predicted output
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = MNIST('.', train=True, download=True, transform=transform)
train_set, val_set = random_split(dataset, [55000, 5000])

train_loader = DataLoader(train_set, batch_size=64, num_workers=0)
val_loader = DataLoader(val_set, batch_size=64, num_workers=0)

# Training
model = MNISTNet()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)
