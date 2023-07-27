#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import covariances as cov
from gpflow.kernels import Kernel
from gpflow.base import TensorLike
from gpflow.inducing_variables import InducingVariables

from sklearn.model_selection import train_test_split

#see https://gpflow.github.io/GPflow/2.4.0/api/gpflow/models/ for details


def train_gp(input, 
             output,
             dim=1,
             kernel=gpflow.kernels.RBF(),
             n_inducing_points = 50,
             noise_variance=0.001,
             test_size=0.2, 
             variational=False,
             epochs=1000,
             ADAM_step=0.01,
             batch_size=100,
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
    
    if not variational:
        GP = gpflow.models.GPR((in_train, out_train), 
                                kernel=kernel, 
                                noise_variance=noise_variance)
        
        
        opt = gpflow.optimizers.Scipy()
        opt.minimize(GP.training_loss, GP.trainable_variables)
    
    else:
        num_latent_gps = out_train.shape[1]
        # data_train = (in_train, out_train)
        N = len(in_train)
        data_train = tf.data.Dataset.from_tensor_slices((in_train, out_train)).repeat().shuffle(N)
        inducing_points = GPInducingVariables(in_train, num_latent_gps)
        
        GP = gpflow.models.SVGP( # model = GraphSVGP(
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(num_latent_gps),
            inducing_variable=inducing_points, #inducing_variable=[0]*num_eigenpairs,
            num_latent_gps=num_latent_gps,
            whiten=True,
            q_diag=True,
        )
        
        logf, GP = run_adam(GP, data_train, epochs, ADAM_step=ADAM_step, batch_size=batch_size)
        
    #test
    out_pred, _ = GP.predict_f(in_test)
    l2_error = np.linalg.norm(out_test - out_pred.numpy(), axis=1).mean()
    print("Relative l2 error is {}".format(l2_error))
    
    return GP


def run_adam(model, data_train, epochs, ADAM_step=0.001, batch_size=100):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param epochs: number of epochs
    """
    logf = []
    train_iter = iter(data_train.batch(batch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(ADAM_step)
    elbo = tf.function(model.elbo)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(epochs):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            print('ELBO loss: {}'.format(elbo))
            logf.append(elbo)
            
    return logf, model


def optimize_model_with_scipy(model, data_train, epochs):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data_train),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": epochs},
    )


class GPInducingVariables(InducingVariables):
    """
       Taken from  2020 Viacheslav Borovitskiy, Iskander Azangulov, 
       Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth, Nicolas Durrande.
       
       Graph inducing points.
       The first coordinate is vertex index.
       Other coordinates are matched with points on \R^d.
       Note that vertex indices are not trainable.
    """
    def __init__(self, x, out_dim):
        self.x_id = x[:, :1]
        if len(x.shape) > 1:
            self.x_v = gpflow.Parameter(x[:, 1:], dtype=gpflow.default_float())

        self.shape_ = list(x.shape)
        print(self.shape_)
        self.shape_.append(out_dim)
        self.shape_ = tf.convert_to_tensor(self.shape_, dtype=tf.int32)

        self.N = self.x_id.shape[0]

    def __len__(self):

        return self.N

    @property
    def GP_IV(self):
        return tf.concat([self.x_id, self.x_v], axis=1)

    @property
    def num_inducing(self) -> tf.Tensor:
        return self.x_id.shape[0]

    @property
    def shape(self):
        return self.shape_

@cov.Kuu.register(GPInducingVariables, gpflow.kernels.Kernel)
def Kuu_kernel_GPinducingvariables(
        inducing_variable: InducingVariables,
        kernel: Kernel,
        jitter=0.0):
    GP_IV = inducing_variable.GP_IV

    Kuu = kernel.K(GP_IV)
    Kuu += jitter * tf.eye(tf.shape(Kuu)[0], dtype=Kuu.dtype)

    return Kuu


@cov.Kuf.register(GPInducingVariables, gpflow.kernels.Kernel, TensorLike)
def Kuf_kernel_GPinducingvariables(
        inducing_variable: InducingVariables,
        kernel: Kernel,
        X: tf.Tensor):
    GP_IV = inducing_variable.GP_IV

    Kuf = kernel.K(GP_IV, X)

    return Kuf



# class GraphSVGP(SVGP):
#     """
#     SVGP via Graph Fourier feature approximations. 
#     See section 3.1 in https://arxiv.org/pdf/2010.15538.pdf.
#     GraphSVGP makes VI for coefficients of eigenvectors of Graph Laplacian
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
#         q_mu = self.q_mu
#         q_sqrt = self.q_sqrt
#         mu, var = self.eval_MV_from_q(q_mu, q_sqrt, Xnew, full_cov)
        
#         return mu + self.mean_function(Xnew), var

#     def eval_MV_from_q(self, q_mu, q_sqrt, X, full_cov=False):
#         """Build the posterior mean and variance from q_mu, q_sqrt"""

#         X_id = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1])
#         S = self.kernel.eval_S(self.kernel.kappa, self.kernel.sigma_f)
#         U = tf.gather_nd(self.kernel.eigenvectors, X_id) * S[None, :]
#         mu = tf.einsum('ij,jl->il', U, q_mu)
        
#         if q_sqrt.shape.ndims == 3:
#             q_cov = tf.einsum('ijn,kjn->ijn', q_sqrt, q_sqrt)
            
#         if q_sqrt.shape.ndims == 2:
#             q_cov = tf.einsum('in,in->in', q_sqrt, q_sqrt)
            
#         if full_cov:
#             if q_sqrt.shape.ndims == 3:
#                 var = tf.einsum('ij,njk,lk->nil', U, q_cov, U)
#             if q_sqrt.shape.ndims == 2:
#                 var = tf.einsum('ij, jn, kj->nik', U, q_cov, U)
#         else:
#             if q_sqrt.shape.ndims == 3:
#                 var = tf.einsum('ij,njk,ik->in', U, q_cov, U)
#             if q_sqrt.shape.ndims == 2:
#                 var = tf.einsum('ij, jn,ij->in', U, q_cov, U)
                
#         return mu, var