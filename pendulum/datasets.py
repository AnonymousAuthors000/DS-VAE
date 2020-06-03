from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
import torch

class SimplePendulumDataset(Dataset):
    def __init__(self, L, thetas_undamped):
        self.thetas_undamped = thetas_undamped
        self.L = L

    def __len__(self):
        return self.thetas_undamped.shape[0]
        
    def __getitem__(self, idx):
        l = self.L[idx]
        theta_undamped = self.thetas_undamped[idx]
        return l, theta_undamped

    
class FrictionPendulumDataset(Dataset):
    def __init__(self, L, B, thetas_damped):
        self.thetas_damped = thetas_damped
        self.L = L
        self.B = B
        
    def __len__(self):
        return self.thetas_damped.shape[0]
        
    def __getitem__(self, idx):
        l = self.L[idx]
        b = self.B[idx]
        theta_damped = self.thetas_damped[idx]
        return l, b, theta_damped
    
    
class MassPendulumDataset(Dataset):
    def __init__(self, L, B, M, theta_damped_mass):
        self.theta_damped_mass = theta_damped_mass
        self.L = L
        self.B = B
        self.M = M
        
    def __len__(self):
        return self.theta_damped_mass.shape[0]
        
    def __getitem__(self, idx):
        l = self.L[idx]
        b = self.B[idx]
        m = self.M[idx]
        theta_damped_mass = self.theta_damped_mass[idx]
        return l, b, m, theta_damped_mass