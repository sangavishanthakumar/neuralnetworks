import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nn_addition.add_model import AdditionDataset, AdditionModel

''' run this file to train the neuronal network '''
''' the results are in the tb_logs folder and can be viewed with tensorboard '''

train_data = AdditionDataset(
    500000)  # amount of samples that should be created are 500000 (sample example: [1,2] label: 3)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # TODO what happens if batch_size is 64?

model = AdditionModel()

logger = TensorBoardLogger('tb_logs', name='addition_model')  # log the results to display them later with tensorboard
trainer = pl.Trainer(max_epochs=2, logger=logger)
trainer.fit(model, train_loader)
