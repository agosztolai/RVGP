#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, LMCVariationalStrategy
import torch
# from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class exact_GP_model(ExactGP):
    def __init__(self, data_train, likelihood, kernel, batch_shape):
        in_train, out_train = data_train
        super().__init__(in_train, out_train, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=batch_shape
        )
        if kernel is None:
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=batch_shape, rank=1
            )
        else:
            self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
  
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.likelihood(self(x))
    
    
class approximate_GP_model(ApproximateGP):
    def __init__(self, inducing_points, kernel=None, batch_shape=1, num_latents=10):
        # Let's use a different set of inducing points for each latent function
        inducing_points = inducing_points.unsqueeze(0).repeat(num_latents, 1, 1)
        
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = LMCVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=batch_shape,
            num_latents=num_latents,
            latent_dim=-1
        )
        
        super(approximate_GP_model, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        if kernel is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents])
            )
        else:
            self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.likelihood(self(x))
        
    
def train_gp(input, 
             output,
             likelihood=None,
             kernel=None,
             dim=1,
             n_inducing_points=10,
             test_size=0.2, 
             variational=True,
             epochs=100,
             lr=0.1,
             batch_size=100,
             seed=0):
    
    #split training and test set
    in_train, in_test, out_train, out_test = \
        train_test_split(input, 
                         output, 
                         test_size=test_size, 
                         random_state=seed
                         )
    
    in_train, in_test, out_train, out_test = \
                    (torch.tensor(in_train, dtype=torch.float32), 
                     torch.tensor(in_test, dtype=torch.float32), 
                     torch.tensor(out_train, dtype=torch.float32), 
                     torch.tensor(out_test, dtype=torch.float32)
                     )
    
    in_train, in_test, out_train, out_test = \
                    (in_train.view(in_train.shape[0]*dim, -1),
                     in_test.view(in_test.shape[0]*dim, -1),
                     out_train.view(out_train.shape[0]*dim, -1),
                     out_test.view(out_test.shape[0]*dim, -1))
    
    if torch.cuda.is_available():
        in_train, out_train, in_test, out_test = \
                    (in_train.cuda(), 
                     out_train.cuda(), 
                     in_test.cuda(), 
                     out_test.cuda()
                     )
    
    num_tasks = out_train.shape[1]
        
    if likelihood is None:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    
    if not variational:
        GP = exact_GP_model((in_train, out_train),
                             likelihood=likelihood,
                             kernel=kernel,
                             batch_shape=out_train.shape[1])
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, GP)
        optimizer = torch.optim.Adam(GP.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
        
    elif variational:
        rng = np.random.default_rng(seed)
        inducing_points = rng.choice(in_train, size=n_inducing_points, replace=False)
        inducing_points = torch.tensor(inducing_points, dtype=torch.float32)
        GP = approximate_GP_model(inducing_points=inducing_points, 
                                  kernel=kernel, 
                                  batch_shape=num_tasks)
        mll = gpytorch.mlls.VariationalELBO(likelihood, GP, num_data=out_train.size(0))
        optimizer = torch.optim.Adam([
                {'params': GP.parameters()},
                {'params': likelihood.parameters()},
                ], lr=lr)
        
        # train_dataset = TensorDataset(in_train, out_train)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # test_dataset = TensorDataset(in_test, out_test)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    if torch.cuda.is_available():
        GP = GP.cuda()
        likelihood = likelihood.cuda()
        
    # Find optimal model hyperparameters
    GP.train()
    likelihood.train()

    epochs_iter = tqdm(range(epochs), desc="Epoch")
    # if not variational:
    for i in epochs_iter:
        optimizer.zero_grad()
        output = GP(in_train)
        loss = -mll(output, out_train)
        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
            
    # elif variational:
    #     epochs_iter = tqdm(range(epochs), desc="Epoch")
    #     for i in epochs_iter:
    #         for x_batch, y_batch in train_loader:
    #             optimizer.zero_grad()
    #             output = GP(x_batch)
    #             loss = -mll(output, y_batch)
    #             epochs_iter.set_postfix(loss=loss.item())
    #             loss.backward()
    #             optimizer.step()
                
    GP.likelihood = likelihood
    
    GP.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        out_pred = GP.predict(in_test)
        l2_error = torch.norm(out_test - out_pred.mean, dim=1).mean()
        print("Relative l2 error is {}".format(l2_error))
    
    return GP
        
    #test
    out_pred, _ = GP.predict_f(in_test)
    l2_error = np.linalg.norm(out_test - out_pred.numpy(), axis=1).mean()
    print("Relative l2 error is {}".format(l2_error))
    
    return GP