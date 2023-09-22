#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from misc import load_mesh
from RVGP.kernels import ManifoldKernel
from RVGP import data, train_gp
from RVGP.geometry import furthest_point_sampling, closest_manifold_point
import polyscope as ps
from tqdm import tqdm

# =============================================================================
# Define manifold points
# =============================================================================
n_eigenpairs=50
n_neighbors=10
vertices, faces = load_mesh('torus')
dim_emb = vertices.shape[1]

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.05)
X = vertices[sample_ind]
d = data(X, faces, n_eigenpairs=n_eigenpairs, n_neighbors=n_neighbors)

d.random_vector_field()
d.smooth_vector_field(t=100)

# =============================================================================
# Train GP for manifold
# =============================================================================
positional_encoding = d.evecs_Lc.reshape(d.n, -1)

manifold_kernel = ManifoldKernel((d.evecs_Lc.reshape(d.n,-1), np.tile(d.evals_Lc, 3)), 
                                  nu=3/2, 
                                  kappa=1.,
                                  typ='matern',
                                  sigma_f=1.)

manifold_GP = train_gp(positional_encoding,
                       X,
                       # n_inducing_points=20,
                       # kernel=manifold_kernel,
                       noise_variance=0.001)

# =============================================================================
# Train GP for predicting spectral coordinates given some real point coordinates
# =============================================================================
vector_field_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                                     nu=3/2, 
                                     kappa=5, 
                                     typ='matern',
                                     sigma_f=1.)

vector_field_GP = train_gp(d.evecs_Lc.reshape(d.n, -1), 
                           d.vectors,
                           dim=d.vertices.shape[1],
                           # n_inducing_points=20,
                           kernel=vector_field_kernel,
                           noise_variance=0.001)

# =============================================================================
# Find trajectory in the vector field
# =============================================================================

# parameters
len_t = 200
h = 0.1 # step size
t0 = 100

y0 = d.vertices[t0, :]
trajectory = [y0]
trajectory_vectors = []

# looping over length of trajectory
for i in tqdm(range(len_t)):
    
    y = trajectory[-1]  
    
    # predict the eigenvectors: TU
    _, x_ = closest_manifold_point(y.reshape(-1,1).T, d)
    
    # predict the manifold point from TU
    y_pred, _ = manifold_GP.predict_f(x_) # should be similar to y
    y_pred = y_pred.numpy()
    
    # predict the vector from the TU
    x_ = x_.reshape(-1, d.evecs_Lc.shape[-1])
    f_pred, f_var = vector_field_GP.predict_f(x_)
    f_pred = f_pred.numpy().reshape(-1, dim_emb)
   
    # perform euler iteration    
    y1 = y_pred + h * f_pred
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    trajectory_vectors.append(f_pred)

trajectory = np.vstack(trajectory)[1:]
trajectory_vectors = np.vstack(trajectory_vectors)

# create plot
ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", d.vertices)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0.0, 0.0, 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", trajectory)
ps_cloud.add_vector_quantity("Predicted vectors", trajectory_vectors, color=(1., 0., 0.), enabled=True)
ps.show()

