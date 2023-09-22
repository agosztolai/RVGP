#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from misc import load_mesh
from RVGP.geometry import furthest_point_sampling, closest_manifold_point
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import polyscope as ps
import numpy as np

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=100
n_neighbors = 10
vertices, faces = load_mesh('sphere')

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.02)
X = vertices[sample_ind]
d = data(X, faces, n_eigenpairs=n_eigenpairs, n_neighbors=n_neighbors)


positional_encoding = d.evecs_Lc.reshape(d.n, -1)

# import numpy as np
# manifold_kernel = ManifoldKernel((d.evecs_Lc.reshape(d.n,-1), np.tile(d.evals_Lc, 3)), 
#                                   nu=3/2, 
#                                   kappa=1., 
#                                   typ='se',
#                                   sigma_f=1.)

manifold_GP = train_gp(positional_encoding,
                        X,
                        # n_inducing_points=20,
                        # kernel=manifold_kernel,
                        epochs=1000,
                        noise_variance=0.001)


np.random.seed(2)
x_query = d.vertices[[0]]+np.random.uniform(size=(1,3))*0.1
x_manifold, pe_manifold = closest_manifold_point(x_query, d)

x_pred, _ = manifold_GP.predict_f(pe_manifold)


ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Off-manifold point", x_query)
ps_cloud = ps.register_point_cloud("Training points", x_manifold)
ps_cloud = ps.register_point_cloud("Predicted points", x_pred)
ps.show()