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
n_eigenpairs=200
n_neighbors=10
vertices, faces = load_mesh('sphere')

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.05)
X = vertices[sample_ind]
d = data(X, faces, n_eigenpairs=n_eigenpairs, n_neighbors=n_neighbors)

# =============================================================================
# Train GP
# =============================================================================
positional_encoding = d.evecs_Lc.reshape(d.n, -1)

import numpy as np
manifold_kernel = ManifoldKernel((d.evecs_Lc.reshape(d.n,-1), np.tile(d.evals_Lc, 3)), 
                                  nu=3/2, 
                                  kappa=1., 
                                  typ='se',
                                  sigma_f=1.)

manifold_GP = train_gp(positional_encoding,
                        X,
                        # n_inducing_points=20,
                        kernel=manifold_kernel,
                        epochs=1000,
                        noise_variance=0.001)

# =============================================================================
# Predict with GP
# =============================================================================
x_test = positional_encoding.reshape(d.n, -1)
y_pred_train, _ = manifold_GP.predict_f(x_test)
y_pred_train = y_pred_train.numpy().reshape(d.n, -1)

# =============================================================================
# Plotting
# =============================================================================
# from RVGP.plotting import graph
# ax = graph(d.G)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection="3d")
# ax.scatter(X[:,0], X[:,1], X[:,2], c='b', s=20, alpha=0.3)
# ax.scatter(y_pred_train[:,0], y_pred_train[:,1], y_pred_train[:,2], c='r', s=20, alpha=0.3)
# ax.axis('equal')

ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", X)
ps_cloud = ps.register_point_cloud("Predicted points", y_pred_train)
ps.show()