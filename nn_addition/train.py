import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from nn_addition.add_model import AdditionDataset, AdditionModel

''' run this file to train the neuronal network '''
''' the results are in the tb_logs folder and can be viewed with tensorboard '''

train_data = AdditionDataset(
    500000)  # amount of samples that should be created are 500000 (sample example: [1,2] label: 3)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # TODO what happens if batch_size is 64?
val_data = AdditionDataset(1000)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
model = AdditionModel()

logger = TensorBoardLogger('tb_logs', name='addition_model')
# log the results to display them later with tensorboard

# early stopping: stop training if the validation loss does not decrease anymore
early_stop_callback = EarlyStopping(
   monitor='val_loss',  # monitor the validation loss
   min_delta=0.00,  # tolerance
   patience=3,  # how many epochs to wait until stopping
   verbose=True,  # print a message if early stopping is triggered
   mode='min'  # min to minimize the loss
)

# enter in terminal (while being in nn_addition/tb_logs/addition_model) the following:
# tensorboard  --logdir version_x
trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=[early_stop_callback])
trainer.fit(model, train_loader, val_loader)

