#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gpflow

from ptu_dijkstra import connections, tangent_frames

from misc import sample_spherical
from RVGP.geometry import (furthest_point_sampling, 
                           manifold_graph, 
                           compute_laplacian,
                           compute_connection_laplacian,
                           compute_spectrum,
                           project_to_manifold, 
                           project_to_local_frame,
                           node_eigencoords
                           )
from RVGP.smoothing import vector_diffusion
from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel

np.random.seed(0)

# =============================================================================
# Define manifold points
# =============================================================================
X = sample_spherical(300)
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.1) #uniform sampling of the datapoints
X = X[sample_ind]
dim_emb = X.shape[1]

# =============================================================================
# Fit graph, tangent frames and connections
# =============================================================================
G = manifold_graph(X)
dim_man = 2
gauges, Sigma = tangent_frames(X, G, dim_man, 10)
R = connections(gauges, G, dim_man)

# =============================================================================
# Eigendecomposition of connection Laplacian
# =============================================================================
L = compute_laplacian(G)
Lc = compute_connection_laplacian(G, R)

n_eigenpairs = 50
evals_Lc, evecs_Lc = compute_spectrum(Lc, n_eigenpairs) # U\Lambda U^T

#rather than U, take TU, where T is the local gauge
evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man,n_eigenpairs)
evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)

# =============================================================================
# Define vector field over manifold and smooth it using vector diffusion
# =============================================================================
f = np.random.uniform(size=(len(X), 3))-.5
f = project_to_manifold(f, gauges[...,:2])
f /= np.linalg.norm(f, axis=1, keepdims=True)


t=100
f = project_to_local_frame(f, gauges)
f = vector_diffusion(f, t, L=L, Lc=Lc, method="matrix_exp")
f = project_to_local_frame(f, gauges, reverse=True)

# =============================================================================
# Custom kernel
# =============================================================================
kernel = ManifoldKernel((evecs_Lc, evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)

# =============================================================================
# Train GP for manifold
# =============================================================================
x_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs) # evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)
y_train = X

manifold_GP = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP.training_loss, manifold_GP.trainable_variables)

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
node_train = np.random.choice(len(X), size=int(len(X)*0.8), replace=False)[:,None]
node_test = np.arange(len(X), dtype=int).reshape(-1, 1)
x_train_f = node_eigencoords(node_train, evecs_Lc, dim_emb)
f_train = f[node_train.flatten()].reshape(-1, 1)
x_test = node_eigencoords(node_test, evecs_Lc, dim_emb)
vector_field_GP = gpflow.models.GPR((x_train_f, f_train), kernel=kernel, noise_variance=0.001)

# optimize_GPR(vector_field_GP, 10000) #this is an alternative using gradient descent
# gpflow.utilities.print_summary(vector_field_GP)    
opt = gpflow.optimizers.Scipy()
opt.minimize(vector_field_GP.training_loss, vector_field_GP.trainable_variables)

# =============================================================================
# Test performance
# =============================================================================
f_pred_mean, _ = vector_field_GP.predict_f(x_test) 
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

l2_error = np.linalg.norm(f.ravel() - f_pred_mean.ravel()) / len(f.ravel())
print("Relative l2 error is {}".format(l2_error))


# =============================================================================
# Train GP for predicting spectral coordinates given some real point coordinates
# =============================================================================

x_train = X
y_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)

manifold_GP_inv = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP_inv.training_loss, manifold_GP_inv.trainable_variables)


# =============================================================================
# Find trajectory in the vector field
# =============================================================================

# parameters
len_t = 100
h = 0.1 # step size
t0 = 100

y0 = x_train[t0, :]
trajectory = [y0]
vectors = []

# looping over length of trajectory
for i in range(len_t):
    
    y = trajectory[-1]  
    
    # predict the eigenvectors: TU
    x_, x_var = manifold_GP_inv.predict_y(y.reshape(-1,1).T)
    
    # predict the manifold point from TU
    y_pred, _ = manifold_GP.predict_f(x_) # should be similar to y
    y_pred = y_pred.numpy()
    
    # predict the vector from the TU
    f_pred, f_var = vector_field_GP.predict_f(x_.numpy().reshape(3,50))
    f_pred = f_pred.numpy().reshape(-1,dim_emb)
   
    # perform euler iteration    
    y1 = y_pred + h * f_pred
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    vectors.append(f_pred)


# create plot
ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)
trajectory = np.vstack(trajectory).T
ax.plot(trajectory[0], trajectory[1], trajectory[2], 'r')



