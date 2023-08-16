import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from nn_addition.add_model import AdditionDataset, AdditionModel

MODEL_PATH = 'tb_logs/addition_model/version_2/checkpoints/epoch=9-step=156250.ckpt'

model = AdditionModel.load_from_checkpoint(MODEL_PATH)  # create an instance of the model

# load_state_dicts expects a dictionary with tensors (weights) and loads them into the model
model.eval()  # set model to evaluation mode

test_data = AdditionDataset(1000)  # create a test dataset with 1000 samples
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)  # create a dataloader for the test dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():  # do not compute gradients because we only evaluate here
    total_loss = 0
    for batch in test_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Average loss: {avg_loss}')
