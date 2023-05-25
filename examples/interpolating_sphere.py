#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gpflow
from scipy import sparse 
import networkx as nx

from ptu_dijkstra import connections, tangent_frames  # isort:skip


from misc import (sample_spherical, 
                  furthest_point_sampling, 
                  manifold_graph, 
                  compute_connection_laplacian,
                  compute_spectrum,
                  sample_from_convex_hull,
                  graph, 
                  )


# =============================================================================
# Define manifold points
# =============================================================================
X = sample_spherical(300)
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.1)#, start_idx=start_idx)
X = X[sample_ind]

# =============================================================================
# Fit graph, tangent frames and connections
# =============================================================================
G = manifold_graph(X)
A = nx.to_scipy_sparse_array(G) + sparse.eye(len(G))
dim_emb = X.shape[1]
dim_man = 2
gauges, Sigma = tangent_frames(X, A, dim_man, 10)
R = connections(gauges, A, dim_man)

# =============================================================================
# Eigendecomposition of connection Laplacian
# =============================================================================
Lc = compute_connection_laplacian(G, R)
n_eigenpairs = 5
_, evecs_Lc = compute_spectrum(Lc, n_eigenpairs)

#rather than U, take TU, where T is the local gauge
evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man, n_eigenpairs)
evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)

# =============================================================================
# Train GP for manifold
# =============================================================================
x_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)
y_train = X
manifold_GP = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP.training_loss, manifold_GP.trainable_variables)

# =============================================================================
# Make predictions
# =============================================================================
x_test = sample_from_convex_hull(x_train, 200, k=3)
y_pred_train, _ = manifold_GP.predict_f(x_test)

ax = graph(G)
ax.scatter(y_pred_train[:,0], y_pred_train[:,1], y_pred_train[:,2], c='r', s=20, alpha=0.3)
ax.axis('equal')
