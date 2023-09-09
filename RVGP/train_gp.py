#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import gpflow
from sklearn.model_selection import train_test_split
from RVGP.geometry import furthest_point_sampling


def train_gp(input, 
             output,
             dim=1,
             n_inducing_points=None,
             test_size=0.2, 
             lr=0.1,
             kernel=None,
             noise_variance=0.001,
             kernel_lengthscale=None,
             kernel_variance=None,
             epochs=1000,
             seed=0,
             compute_error=True):
    
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
    
    if kernel is None:
        kernel = gpflow.kernels.RBF()
    
    if n_inducing_points is None:
        GP = gpflow.models.GPR((in_train, out_train), 
                                kernel=kernel, 
                                noise_variance=noise_variance,
                                )
    else:
        ind, _ = furthest_point_sampling(in_train, N=n_inducing_points)
        inducing_variable = in_train[ind]
        
        GP = gpflow.models.SGPR((in_train, out_train), 
                                kernel=kernel, 
                                noise_variance=noise_variance,
                                inducing_variable=inducing_variable)
        
    if kernel_variance is not None:
        kernel.variance.assign(kernel_variance)
        gpflow.set_trainable(kernel.variance, False)
        
    if kernel_lengthscale is not None:
        kernel.lengthscales.assign(kernel_lengthscale)
        gpflow.set_trainable(kernel.lengthscales, False)
        
    # logf, GP = run_adam(GP, epochs, lr=lr)
    GP = optimize_model_with_scipy(GP, epochs)

    #test
    if compute_error:
        out_pred, _ = GP.predict_f(in_test)
        l2_error = np.linalg.norm(out_test - out_pred.numpy(), axis=1).mean()    
        print("Relative l2 error is {}".format(l2_error))
        
    return GP


def run_adam(model, epochs, lr=0.001):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param epochs: number of epochs
    """
    logf = []
    training_loss = model.training_loss
    optimizer = tf.optimizers.Adam(lr)
    
    if hasattr(model, 'log_marginal_likelihood'):
        loss_name = 'log_marginal_likelihood'
        # loss = tf.function(model.log_marginal_likelihood)
    elif hasattr(model, 'elbo'):
        loss_name = 'ELBO'
        # loss = tf.function(model.elbo)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(epochs):
        optimization_step()
        if step % 10 == 0:
            loss = -training_loss().numpy()
            print(loss_name + ' : {}'.format(loss))
            logf.append(loss)
            
    return logf, model


def optimize_model_with_scipy(model, epochs):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss,
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": epochs},
    )
    return model