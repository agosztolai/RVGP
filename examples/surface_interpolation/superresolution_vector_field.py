#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import polyscope as ps
import numpy as np
from sklearn.metrics import pairwise_distances

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=100
n_neighbors=10
vertices, faces = load_mesh('bunny') #see /examples/data for more objects

# =============================================================================
# Subsample 
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.03)
X = vertices[sample_ind]

train_ind =  np.random.choice(np.arange(len(X)), size=int(0.5*len(X)))
test_ind = [i for i in range(len(X)) if i not in train_ind]
# =============================================================================
# Add noise
# =============================================================================
diam = pairwise_distances(vertices).max()
X[test_ind] += 0.04*diam*np.random.normal(size=X[test_ind].shape)

# =============================================================================
# Create data object
# =============================================================================
d = data(X, faces, n_eigenpairs=n_eigenpairs,dim_man=3)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)

# =============================================================================
# Superresolution
# =============================================================================
train_x, train_y, train_f = d.evecs_Lc.reshape(d.n, -1)[train_ind], X[train_ind], d.vectors[train_ind]
test_x, test_y, test_f = d.evecs_Lc.reshape(d.n, -1)[test_ind], X[test_ind], d.vectors[test_ind]

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