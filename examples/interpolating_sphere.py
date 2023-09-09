#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from RVGP.plotting import graph
from misc import load_mesh
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel



vertices, faces = load_mesh('sphere')

n_eigenpairs=20
d = data(vertices, faces, n_eigenpairs=n_eigenpairs)

# =============================================================================
# Graph laplacian
# =============================================================================
positional_encoding = d.evecs_L.reshape(d.n, -1)

manifold_kernel = ManifoldKernel((d.evecs_L, d.evals_L), 
                        nu=3/2, 
                        kappa=5, 
                        typ='matern',
                        sigma_f=1.)

manifold_GP = train_gp(positional_encoding,
                        d.vertices,
                              # n_inducing_points=20,
                                kernel=manifold_kernel,
                                # kernel_variance=20.,
                                # kernel_lengthscale=2,
                              epochs=1000,
                              noise_variance=0.001)

x_test = positional_encoding.reshape(-1, n_eigenpairs)
y_pred_train, _ = manifold_GP.predict_f(x_test)
y_pred_train = y_pred_train.numpy().reshape(d.n, -1)

ax = graph(d.G)
ax.scatter(y_pred_train[:,0], y_pred_train[:,1], y_pred_train[:,2], c='r', s=20, alpha=0.3)
ax.axis('equal')

# =============================================================================
# COnnection laplacian
# =============================================================================
positional_encoding = d.evecs_Lc.reshape(d.n, -1)

manifold_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        typ='se',
                        sigma_f=1.)

manifold_GP = train_gp(positional_encoding,
                        d.vertices,
                              # n_inducing_points=20,
                                # kernel=manifold_kernel,
                                # dim=3,
                                # kernel_variance=20.,
                                # kernel_lengthscale=2,
                              epochs=1000,
                              noise_variance=0.001)

x_test = positional_encoding#.reshape(-1, n_eigenpairs)
y_pred_train, _ = manifold_GP.predict_f(x_test)
y_pred_train = y_pred_train.numpy().reshape(d.n, -1)

ax = graph(d.G)
ax.scatter(y_pred_train[:,0], y_pred_train[:,1], y_pred_train[:,2], c='r', s=20, alpha=0.3)
ax.axis('equal')
