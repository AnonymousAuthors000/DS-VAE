import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import pytorch_lightning as pl
import os
import numpy as np
import math

class MassVAE(pl.LightningModule):
    def __init__(self, z_size, output_size, train_loader, friction_model, learning_rate=1e-3, beta1=.9, beta2=.999, likelihood_var=.25, batch_norm=False):
        super(MassVAE, self).__init__()
        friction_model.freeze()
        self.friction_model = friction_model
        self.z_size = z_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.beta1 = beta1
        self.beta2 = beta2
        self.likelihood_var = likelihood_var
        self.batch_norm = batch_norm
    
        self.fc1_enc = torch.nn.Linear(output_size, 400)  
        if batch_norm:
            self.batch_norm_1 = torch.nn.BatchNorm1d(400)
            
        self.fc_mu = torch.nn.Linear(400, z_size)
        self.fc_var = torch.nn.Linear(400, z_size)
        self.fc1_dec = torch.nn.Linear(z_size + output_size, 400)        

        self.fc2_dec = torch.nn.Linear(400, 400)
        self.theta1_layer = torch.nn.Linear(400, output_size)   # reconstruct angular displacement
         
    def encode(self, residual):
        EPSILON = 1e-6
        if self.batch_norm:
            h = F.relu(self.batch_norm_1(self.fc1_enc(residual)))
        else:
            h =  F.relu(self.fc1_enc(residual))
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h)) + EPSILON
        return mu, var

    def reparameterize_normal(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def decode(self, theta_damped, z):
        x = torch.cat((theta_damped, z), axis=1)
        h = torch.sigmoid(self.fc1_dec(x))
        h = torch.sigmoid(self.fc2_dec(h))
        theta1 = self.theta1_layer(h) 
        return theta1
    
    def forward(self, l, b, theta_damped_mass):
        theta_damped = self.friction_model(l, b)
        residual = theta_damped_mass - theta_damped 
        mu, var = self.encode(residual)
        if self.training:
            z = self.reparameterize_normal(mu, var)
        else:
            z = mu
        recon_thetas_damped_mass = self.decode(theta_damped, z)
        return recon_thetas_damped_mass, mu, var
    
    def training_step(self, batch, batch_idx):
        # REQUIRED
        l, b, m, theta_damped_mass = batch
        # generate thetas_undamped from previously trained model
        recon_thetas_damped_mass, mu, var = self.forward(l, b, theta_damped_mass)
        negative_elbo, nll, kl = negative_ELBO(recon_thetas_damped_mass, theta_damped_mass, mu, var, self.likelihood_var) 
        logs = {'m_negative_elbo': negative_elbo, 'm_nll': nll, 'm_kld': kl}
        return {'loss': negative_elbo, 'progress_bar': {'m_nll': nll, 'm_kld': kl}, 'log': logs}
    
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.train_loader
    
    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

    def on_train_end(self):
        if os.path.exists('saved_models'):
            torch.save({
                'model_state_dict': self.state_dict(),
                'z_size': self.z_size,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'batch_norm': self.batch_norm,
                }, "saved_models/mass_vae")

# Reconstruction + KL divergence losses summed over all elements and batch
def negative_ELBO(recon_X, X, mu, var, likelihood_var):
    device = recon_X.get_device()
    batch_size = recon_X.shape[0]
    normal = torch.distributions.Normal(X,likelihood_var)
    NLL = -1*torch.sum(normal.log_prob(recon_X))/batch_size
    
    m_q = mu
    m_p = torch.zeros(m_q.shape).to(device)
    var_q = var
    var_p = (torch.zeros(var_q.shape) + 1**2).to(device)
    KLD = kl_divergence_normal(m_q, var_q, m_p, var_p)/batch_size
   
    return NLL + KLD, NLL, KLD

def kl_divergence_normal(mu_q, var_q, mu_p, var_p):
    kld = torch.sum(0.5*(torch.log(var_p) - torch.log(var_q)) + torch.div(var_q + (mu_q - mu_p)**2, 2*var_p) - 0.5)
    return kld
