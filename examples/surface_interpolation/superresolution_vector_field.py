#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from misc import load_mesh
from RVGP.geometry import furthest_point_sampling
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import polyscope as ps
import numpy as np

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=100
n_neighbors=10
vertices, faces = load_mesh('bunny')

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.01)
X = vertices[sample_ind]
d = data(X, faces, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)

# =============================================================================
# Superresolution
# =============================================================================
train_ind =  np.random.choice(np.arange(len(X)),size=int(0.5*len(X)))
train_x, train_y, train_f = d.evecs_Lc.reshape(d.n, -1)[train_ind], X[train_ind], d.vectors[train_ind]
test_x, test_y, test_f = d.evecs_Lc.reshape(d.n, -1), X, d.vectors

# =============================================================================
# Mask out part of the vector field
# =============================================================================
# singularity = np.array([-0.08679473,  0.1146,  0.0022]) 
# train_ind =  np.linalg.norm(X-singularity, axis=1) > 0.015
# train_x, train_y, train_f = d.evecs_Lc.reshape(d.n, -1)[train_ind], X[train_ind], d.vectors[train_ind]
# test_x, test_y, test_f = d.evecs_Lc.reshape(d.n, -1)[~train_ind], X[~train_ind], d.vectors[~train_ind]

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
vector_field_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                                     nu=3/2, 
                                     kappa=5, 
                                     typ='matern',
                                     sigma_f=1.)

vector_field_GP = train_gp(train_x,
                           train_f,
                           dim=vertices.shape[1],
                           kernel=vector_field_kernel,
                           noise_variance=0.001)

# =============================================================================
# Predict with GPs
# =============================================================================
n = len(test_x)
test_x = test_x.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(test_x)
f_pred_mean = f_pred_mean.numpy().reshape(n, -1)


ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", train_y)
ps_cloud.add_vector_quantity("Training vectors", d.vectors[train_ind], color=(227, 156, 28), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", test_y)
ps_cloud.add_vector_quantity("Predicted vectors", f_pred_mean, color=(227, 28, 28),  enabled=True)
ps.show()