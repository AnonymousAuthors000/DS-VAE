import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import os
import pytorch_lightning as pl

class SimulatorModel(pl.LightningModule):
    def __init__(self, input_size, output_size, train_loader, learning_rate=1e-3):
        super(SimulatorModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.loss_func = torch.nn.MSELoss()
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        
        self.fc1 = torch.nn.Linear(input_size, 400)
        self.fc2 = torch.nn.Linear(400, 400)
        self.theta1_layer = torch.nn.Linear(400, output_size)   # reconstruct angular displacement

    def forward(self, l):
        h = torch.sigmoid(self.fc1(l))
        h = torch.sigmoid(self.fc2(h))
        theta1 = self.theta1_layer(h)
        return theta1
    
    def training_step(self, batch, batch_idx):
        # REQUIRED
        l, theta_undamped = batch
        theta_undamped_hat = self.forward(l)
        loss = self.loss_func(theta_undamped_hat, theta_undamped)
        logs = {'s_train_loss': loss}
        return {'loss': loss, 'log': logs}
    
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.train_loader
    
    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_train_end(self):
        if os.path.exists('saved_models'):
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate,
                }, "saved_models/simulator_model")