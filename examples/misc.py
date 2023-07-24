#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


# def posterior(kernel, X_s, X_train, Y_train, sigma_y=1e-8):
#     """
#     Computes the suffifient statistics of the posterior distribution 
#     from m training data X_train and Y_train and n new inputs X_s.
    
#     Args:
#         X_s: New input locations (n x d).
#         X_train: Training locations (m x d).
#         Y_train: Training targets (m x 1).
#         l: Kernel length parameter.
#         sigma_f: Kernel vertical variation parameter.
#         sigma_y: Noise parameter.
    
#     Returns:
#         Posterior mean vector (n x d) and covariance matrix (n x n).
#     """
        
#     K = kernel.K(X_train, X_train).numpy()
#     K += sigma_y**2 * np.eye(len(K))
#     K_s = kernel.K(X_train, X_s).numpy()
#     K_ss = kernel.K(X_s, X_s).numpy() 
#     K_ss += 1e-8 * np.eye(len(K_ss))
#     K_inv = inv(K)
    
#     mu_s = K_s.T.dot(K_inv).dot(Y_train)
#     cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
#     return mu_s, cov_s