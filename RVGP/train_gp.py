#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gpflow
import numpy as np
from sklearn.model_selection import train_test_split


def train_gp(input, 
             output,
             dim=1,
             kernel=gpflow.kernels.RBF(),
             noise_variance=0.001,
             test_size=0.2, 
             seed=0):
    
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
    
    #set up optimiser
    GP = gpflow.models.GPR((in_train, out_train), 
                           kernel=kernel, 
                           noise_variance=noise_variance)

    #optimise
    opt = gpflow.optimizers.Scipy()
    opt.minimize(GP.training_loss, GP.trainable_variables)
    
    #test
    out_pred, _ = GP.predict_f(in_test)
    l2_error = np.linalg.norm(out_test - out_pred.numpy(), axis=1).mean()
    print("Relative l2 error is {}".format(l2_error))
    
    return GP