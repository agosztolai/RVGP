"""RNN definition of network classes, training functionality."""
import os
import torch.nn as nn
from math import sqrt, floor
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn

from .RNN_helpers import move_to_gpu, timeseries

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
    
def train(net, x, inp=None, epochs=100, batch_size=32, lr=0.001, outdir=None):
    
    net, x, inp = move_to_gpu(net, x, inp)
    
    dataset = timeseries(x, inp)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    net._epoch = 0
    best_loss = -1
    for epoch in range(epochs):
        net._epoch = epoch
        training_loss = 0
        for batch in train_loader:
            if inp is not None:
                x, inp = batch 
            else:
                x = batch 
            
            # Forward pass
            x_pred, x_reconstruct, _ = net(x, seq_len=x.shape[1], inp=inp)
                                    
            # Compute training loss
            loss = net.loss_fun(x, x_reconstruct, x_pred)
            training_loss += loss
            # scheduler.step(loss)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss /= len(train_loader)
            
        val_loss = 0
        for batch in val_loader:
            if inp is not None:
                x, inp = batch 
            else:
                x = batch 
            
            # Forward pass
            x_pred, x_reconstruct, _ = net(x, seq_len=x.shape[1], inp=inp)
                                    
            # Compute validation loss
            loss = net.loss_fun(x, x_reconstruct, x_pred)
            val_loss += loss
             
        val_loss /= len(val_loader)
        
        # Print the loss for every few epochs
        if epoch%10==0:
            print(f"\nEpoch [{epoch+1}/{epochs}], Training loss: {training_loss.item()}, Validation loss: {val_loss.item()}", end="")
            
            if best_loss == -1 or (loss < best_loss):
                outdir = save_model(net, optimizer, outdir, best=True, timestamp=time)
                best_loss = val_loss
                print(" *", end="")
            
    save_model(net, optimizer, outdir, best=False, timestamp=time)
    load_model(net, os.path.join(outdir, f"best_model_{time}.pth"))
            
            
def save_model(net, optimizer, outdir=None, best=False, timestamp=""):
        """Save model."""
        if outdir is None:
            outdir = "./outputs/"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        checkpoint = {
            "epoch": net._epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "time": timestamp,
        }

        if best:
            fname = "best_model_"
        else:
            fname = "last_model_"

        fname += timestamp
        fname += ".pth"

        if best:
            torch.save(checkpoint, os.path.join(outdir, fname))
        else:
            torch.save(checkpoint, os.path.join(outdir, fname))

        return outdir
    
    
def load_model(net, loadpath):
        """Load model.

        Args:
            loadpath: directory with models to load best model, or specific model path
        """
        checkpoint = torch.load(loadpath)
        net._epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["model_state_dict"])
        net.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        
        return net
                
                
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        
        assert isinstance(hidden_dim, list)
        
        self.encoder = nn.Sequential()
        for i, d in enumerate(hidden_dim):
            if i==0:
                self.encoder.append(nn.Linear(input_dim, d))
            else:
                self.encoder.append(nn.ReLU())
                self.encoder.append(nn.Linear(hidden_dim[i-1],d))
        
        hidden_dim = hidden_dim[::-1]
        self.decoder = nn.Sequential()
        for i, d in enumerate(hidden_dim):
            if i==len(hidden_dim)-1:
                self.decoder.append(nn.Linear(d, input_dim))
            else:
                self.decoder.append(nn.Linear(d, hidden_dim[i+1]))
                self.decoder.append(nn.ReLU())
        
    def forward(self, x):
        
        mask = torch.isnan(x)
        x[mask] = 0.0
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class LatentLowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rank=2, noise_std=5e-2, alpha=0.2):
        super(LatentLowRankRNN, self).__init__()
        
        self.autoencoder = Autoencoder(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.non_linearity = torch.tanh
        self.MSELoss = nn.MSELoss()
        
        #hyperparameters
        self.alpha = alpha
        self.noise_std = noise_std
        self.rank = rank
        self.input_size = input_size
        self.hidden_size = hidden_size[-1]
        
        #trainable parameters
        self.m = nn.Parameter(torch.Tensor(self.hidden_size, rank))
        self.n = nn.Parameter(torch.Tensor(self.hidden_size, rank))
        # self.svd_reparametrization()
        self.wi = nn.Parameter(torch.Tensor(rank, self.hidden_size))
        self.lambda1 = nn.Parameter(torch.tensor(float()))
        
        with torch.no_grad():
            self.m.normal_()
            self.n.normal_()
            self.wi.normal_()
            self.lambda1.normal_()
            
    def loss_fun(self, x, x_reconstruct, x_pred):
        
        mask = torch.isnan(x)
        x, x_reconstruct, x_pred = x[~mask], x_reconstruct[~mask], x_pred[~mask],
        
        #autoencoder loss
        loss = self.MSELoss(x, x_reconstruct)
        
        with torch.no_grad():
            self.lambda1.data = torch.clamp(self.lambda1, min=1e-8)
            
        #prediction loss
        loss += self.MSELoss(x, x_pred)
        
        return loss
            
#     def forward(self, x, dx, inp):
#         z, x_reconstruct = self.autoencoder(x)
        
#         dz = self.derivative(x, dx, self.autoencoder.encoder[::2])
#         dz_pred = self.LowRankRNN(z, inp=inp)
#         dx_pred = self.derivative(z, dz_pred, self.autoencoder.decoder[::2])
        
#         return x_reconstruct, dz, dz_pred, dx_pred
    
    def forward(self, x, seq_len=100, inp=None):
        """
        :param x: tensor of shape (batch_size, timepoints, input_dimension)
        Important: the 2 dimensions need to be present, even if they are of size 1.
        """
        
        assert len(x.shape)==3, 'x must be of shape trials x dim. For initial conditions, timepoinst=1'
        
        x0 = x[:,0,:]
        
        h, x_reconstruct = self.autoencoder(x)
        h = h[:,0,:].clone()
        
        batch_size = len(x0)
        x_pred = torch.zeros(batch_size, seq_len, self.input_size, device=self.m.device)
        z_pred = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.m.device)
        
        x_pred[:, 0, :], z_pred[:, 0, :] = x0, h
        for i in range(seq_len-1):
            dh = self.LowRankRNN(h[:,None,:], inp)
            h += dh
            z_pred[:, i + 1, :] = h
            r = self.non_linearity(h)
            #x_pred[:, i + 1, :] = self.autoencoder.decoder(r)
            out = self.autoencoder.decoder(r)
            out[:, 0] = out[:, 0] % (2 * torch.pi) 

            x_pred[:, i + 1, :] = self.autoencoder.decoder(r)
            
        return  x_pred, x_reconstruct, z_pred
    
    def derivative(self, x, dx, layers):
        for l in layers[:-1]:
            x = l(x)
            dx = torch.multiply((x>0).float(), torch.matmul(dx, l.weight.T))
            x = self.relu(x)

        return torch.matmul(dx, layers[-1].weight.T)
    
    def LowRankRNN(self, z, inp=None):
        
        assert len(z.shape)==3

        batch_size, seq_len, _ = z.shape
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        dz = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.m.device)
        
        for i in range(seq_len):
            z_, noise_ = z[:,i,:], noise[:,i,:]
            r_ = self.non_linearity(z_)
            dz_ = self.alpha * ( -z_ + r_.matmul(self.n).matmul(self.m.t()) / self.hidden_size ) \
                                       + self.noise_std * noise_
            if inp is not None:
                inp_ = inp[:,i,:]
                dz_ += self.alpha * inp_.matmul(self.wi)
                
            dz[:,i,:] = dz_
        return dz.squeeze()
    
    
    def svd_reparametrization(self):
        """Orthogonalize m and n"""
        with torch.no_grad():
            structure = (self.m @ self.n.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n.set_(torch.from_numpy(n.T * np.sqrt(s)))