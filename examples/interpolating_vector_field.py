#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from misc import load_mesh
from RVGP.geometry import furthest_point_sampling
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import polyscope as ps

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=100
n_neighbors=10
vertices, faces = load_mesh('bunny')

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.015)
X = vertices[sample_ind]
d = data(X, faces, n_eigenpairs=n_eigenpairs)

# =============================================================================
# Mask out part of the vector field
# =============================================================================
#this is for 'bunny'
import numpy as np
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)
singularity = np.array([-0.08679473,  0.1146,  0.0022])
train_ind =  np.linalg.norm(X-singularity, axis=1) > 0.015
train_x, train_y = d.evecs_Lc.reshape(d.n, -1)[train_ind], X[train_ind]
test_x, test_y = d.evecs_Lc.reshape(d.n, -1)[~train_ind], X[~train_ind]

# =============================================================================
# Train GP for manifold
# =============================================================================
# positional_encoding = d.evecs_Lc.reshape(d.n, -1)

manifold_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                                  nu=3/2, 
                                  kappa=5, 
                                  typ='matern',
                                  sigma_f=1.)

# manifold_GP = train_gp(positional_encoding,
#                        X,
#                        # n_inducing_points=20,
#                        # kernel=manifold_kernel,
#                        epochs=1000,
#                        noise_variance=0.001)

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
vector_field_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                                     nu=3/2, 
                                     kappa=5, 
                                     typ='matern',
                                     sigma_f=1.)

vector_field_GP = train_gp(d.evecs_Lc.reshape(d.n, -1),
                            d.vectors,
                            dim=vertices.shape[1],
                            epochs=1000,
                            # n_inducing_points=20,
                            kernel=vector_field_kernel,
                            noise_variance=0.001)

# =============================================================================
# Predict with GPs
# =============================================================================
x_test = test_x#d.evecs_Lc.reshape(d.n, -1)
# y_pred_mean, _ = manifold_GP.predict_f(x_test)

n = len(test_x)
x_test = x_test.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(x_test)
f_pred_mean = f_pred_mean.numpy().reshape(n, -1)

# =============================================================================
# Plotting
# =============================================================================
# from RVGP.plotting import graph
# ax = graph(d.G)
# ax.quiver(X[:,0], X[:,1], X[:,2], d.vectors[:,0], d.vectors[:,1], d.vectors[:,2], color='g', length=0.3)
# ax.quiver(y_pred_mean[:,0], y_pred_mean[:,1], y_pred_mean[:,2], f_pred_mean[:,0], f_pred_mean[:,1], f_pred_mean[:,2], color='r', length=0.3)
# ax.axis('equal')

ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", train_y)
ps_cloud.add_vector_quantity("Training vectors", d.vectors[train_ind], color=(0., 0., 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", test_y)
ps_cloud.add_vector_quantity("Predicted vectors", f_pred_mean, color=(1., 0., 0.), enabled=True)
ps.show()