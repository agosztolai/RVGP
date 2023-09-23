#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import polyscope as ps
import numpy as np

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=300
n_neighbors=10
vertices, faces = load_mesh('torus')

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.04)
X = vertices[sample_ind]
d = data(X, faces, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)


positional_encoding = d.evecs_Lc.reshape(d.n, -1)

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
vector_field_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                                     nu=3/2, 
                                     kappa=5, 
                                     typ='matern',
                                     sigma_f=1.)

vector_field_GP = train_gp(positional_encoding,
                           d.vectors,
                           dim=vertices.shape[1],
                           kernel=vector_field_kernel,
                           noise_variance=0.001)

# =============================================================================
# Predict with GPs
# =============================================================================
n = len(positional_encoding)
test_x = positional_encoding.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(test_x)
f_pred_mean = f_pred_mean.numpy().reshape(n, -1)


ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", d.vertices)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(227, 156, 28), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", d.vertices)
ps_cloud.add_vector_quantity("Predicted vectors", f_pred_mean, color=(227, 28, 28),  enabled=True)
ps.show()