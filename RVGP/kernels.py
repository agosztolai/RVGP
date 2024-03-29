#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import gpflow

class ManifoldKernel(gpflow.kernels.Kernel):
    """Matern kernel on manifold. 

    Attributes
    ----------
    data : data object
    typ : str
        Type of kernel. 'matern' or 'SE'
    nu : float
        Trainable smoothness hyperparameter.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    dtype : tf.dtypes.DType
        type of tensors, tf.float64 by default
        """

    def __init__(self, data, nu=3, kappa=4, sigma_f=1, typ='matern', dtype=tf.float64):

        self.eigenvectors, self.eigenvalues = data.evecs_Lc, data.evals_Lc
        self.num_verticies = tf.cast(tf.shape(self.eigenvectors)[0], dtype=dtype)
        self.dtype = dtype
        self.typ = typ
        
        if typ not in ('se', 'matern'):
            NotImplemented
        
        if typ == 'matern':
            self.nu = gpflow.Parameter(nu, dtype=self.dtype, transform=gpflow.utilities.positive(), name='nu')
        self.kappa = gpflow.Parameter(kappa, dtype=self.dtype, transform=gpflow.utilities.positive(), name='kappa')
        self.sigma_f = gpflow.Parameter(sigma_f, dtype=self.dtype, transform=gpflow.utilities.positive(), name='sigma_f')
        super().__init__()

    def eval_S(self, typ = 'matern'):
        """Wilson Eq. (69)"""
        if typ == 'matern':
            S = tf.pow(self.eigenvalues + 2*self.nu/(self.kappa**2), -self.nu)
            S = tf.multiply(S, self.num_verticies/tf.reduce_sum(S))
            S = tf.multiply(S, self.sigma_f)
            
        elif typ == 'se':
            S = tf.exp(-0.5*self.eigenvalues*self.kappa**2)
            S = tf.multiply(S, self.num_verticies/tf.reduce_sum(S))
            S = tf.multiply(S, self.sigma_f)
        
        return S

    def K(self, X, X2=None):
        """Kernel function"""
        if X2 is None:
            X2 = X
            
        S = self.eval_S(typ=self.typ)
        return (X * S) @ tf.transpose(X2) # shape (n,n)

    def K_diag(self, X):
        """This is just the diagonal of K"""
        
        S = self.eval_S(typ=self.typ)
        return tf.linalg.tensor_diag_part((X * S) @ tf.transpose(X))
    