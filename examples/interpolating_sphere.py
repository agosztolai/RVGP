#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from RVGP.plotting import graph
from misc import load_mesh
from RVGP.geometry import furthest_point_sampling
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel

import polyscope as ps


vertices, faces = load_mesh('monkey')

sample_ind, _ = furthest_point_sampling(vertices, stop_crit=0.015)#, start_idx=start_idx)
X = vertices[sample_ind]

n_eigenpairs=20
n_neighbors=10
d = data(X, faces, n_eigenpairs=n_eigenpairs, n_neighbors=n_neighbors)

# =============================================================================
# Graph laplacian
# =============================================================================
positional_encoding = d.evecs_Lc.reshape(d.n, -1)

manifold_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        typ='matern',
                        sigma_f=1.)

manifold_GP = train_gp(positional_encoding,
                        X,
                        # dim=3,
                              # n_inducing_points=20,
                                # kernel=manifold_kernel,
                                # kernel_variance=20.,
                                # kernel_lengthscale=2,
                              epochs=1000,
                              noise_variance=0.001)

x_test = positional_encoding.reshape(d.n, -1)
y_pred_train, _ = manifold_GP.predict_f(x_test)
y_pred_train = y_pred_train.numpy().reshape(d.n, -1)

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