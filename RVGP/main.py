#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gpflow
from sklearn.model_selection import train_test_split
from RVGP.geometry import furthest_point_sampling
from RVGP.kernels import ManifoldKernel


def train_gp(data,
             train_ind=None,
             n_inducing_points=None,
             test_size=0.2, 
             kernel=None,
             noise_variance=0.001,
             kernel_lengthscale=None,
             kernel_variance=None,
             epochs=1000,
             positivity_constraint=1e-2,
             seed=0):
    
    
    if train_ind is None:
        train_ind = np.arange(data.n)
    
    output = data.vectors[train_ind]
    
    if kernel is None:
        kernel = ManifoldKernel(data, nu=3/2, kappa=5, typ='matern', sigma_f=1.)
        input = data.evecs_Lc.reshape(data.n, -1)[train_ind]
        dim = output.shape[1]
    if kernel=='rbf':
        kernel = gpflow.kernels.RBF()
        input = data.evecs_L.reshape(data.n, -1)[train_ind]
        dim = 1
        print('Using RBF kernel, treating vectors channel-wise.')
    
    #split training and test set
    in_train, in_test, out_train, out_test = \
        train_test_split(input, 
                         output, 
                         test_size=test_size, 
                         random_state=seed
                         )
        
    in_train = in_train.reshape(in_train.shape[0]*dim, -1)
    in_test = in_test.reshape(in_test.shape[0]*dim, -1)
    out_train = out_train.reshape(out_train.shape[0]*dim, -1)
    out_test = out_test.reshape(out_test.shape[0]*dim, -1)
    
    gpflow.config.set_default_positive_minimum(positivity_constraint)
    
    if n_inducing_points is None:
        GP = manifold_GPR((in_train, out_train), 
                          kernel, 
                          noise_variance=noise_variance,
                          )
    else:
        ind, _ = furthest_point_sampling(in_train, N=n_inducing_points)
        inducing_variable = in_train[ind]
        
        GP = manifold_SGPR((in_train, out_train), 
                           kernel,
                           inducing_variable,
                           noise_variance=noise_variance
                           )
    
    if kernel_variance is not None:
        kernel.variance.assign(kernel_variance)
        gpflow.set_trainable(kernel.variance, False)
        
    if kernel_lengthscale is not None:
        kernel.lengthscales.assign(kernel_lengthscale)
        gpflow.set_trainable(kernel.lengthscales, False)
        
    GP = optimize_model_with_scipy(GP, epochs)

    #test
    out_pred, _ = GP.predict_f(in_test)
    l2_error = np.linalg.norm(out_test - out_pred.numpy(), axis=1).mean()    
    print("Relative l2 error is {}".format(l2_error))
        
    return GP


def optimize_model_with_scipy(model, epochs):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss,
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": epochs},
    )
    return model


class manifold_GPR(gpflow.models.GPR):
    def __init__(self, data, kernel, mean_function=None, noise_variance=None, likelihood=None):
        super().__init__(data, kernel, mean_function=None, noise_variance=None, likelihood=None)
        
    def transform(self, data, test_ind):
        
        if isinstance(test_ind[0], int):
            test_x = data.evecs_Lc.reshape(data.n, -1)[test_ind]
            test_x = test_x.reshape(-1, data.evecs_Lc.shape[1])
            
        elif isinstance(test_ind[0], float):
            test_x = test_ind
        
        f_pred_mean, f_pred_std = self.predict_f(test_x)
        
        f_pred_mean = f_pred_mean.numpy().reshape(len(test_ind), -1)
        f_pred_std = f_pred_std.numpy().reshape(len(test_ind), -1)
        
        return f_pred_mean, f_pred_std
    

class manifold_SGPR(gpflow.models.SGPR):
    def __init__(self, data, kernel, inducing_variable, mean_function=None, noise_variance=None, likelihood=None):
        super().__init__(data, kernel, inducing_variable, mean_function=None, noise_variance=None, likelihood=None)
        
    def transform(self, data, test_ind):
        
        if isinstance(test_ind[0], int):
            test_x = data.evecs_Lc.reshape(data.n, -1)[test_ind]
            test_x = test_x.reshape(-1, data.evecs_Lc.shape[1])
            
        elif isinstance(test_ind[0], float):
            test_x = test_ind
        
        f_pred_mean, f_pred_std = self.predict_f(test_x)
        
        f_pred_mean = f_pred_mean.numpy().reshape(len(test_ind), -1)
        f_pred_std = f_pred_std.numpy().reshape(len(test_ind), -1)
        
        return f_pred_mean, f_pred_std