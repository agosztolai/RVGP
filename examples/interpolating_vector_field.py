#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from misc import load_mesh
from RVGP.kernels import ManifoldKernel
from RVGP import data, train_gp
import polyscope as ps

# =============================================================================
# Define manifold points
# =============================================================================
X, faces = load_mesh('sphere')

n_eigenpairs=10
d = data(X, faces, n_eigenpairs=n_eigenpairs)

d.random_vector_field()
d.smooth_vector_field(t=100)

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# pca = PCA(n_components=2)
# emb = pca.fit_transform(d.evecs_L)
# plt.scatter(emb[:,0], emb[:,1])

# =============================================================================
# points to eliminate around singularity, ok as long as seed is the same
# =============================================================================
import numpy as np
radius=0.2
singularity = np.array([-0.672737181, 0.415925592, -0.491816372])
train_ind = np.linalg.norm(d.vertices-singularity, axis=1) > radius
train_ind = np.ones(d.n, dtype=bool)
test_ind = train_ind

# =============================================================================
# Train GP for manifold
# =============================================================================
manifold_kernel = ManifoldKernel((d.evecs_L, d.evals_L), 
                        nu=3/2, 
                        kappa=5, 
                        typ='se',
                        sigma_f=1.)

manifold_GP = train_gp(d.evecs_L.reshape(d.n, -1)[train_ind],
                              d.vertices[train_ind],
                              # n_inducing_points=20,
                               # kernel=manifold_kernel,
                               kernel_variance=1.,
                              epochs=1000,
                              noise_variance=0.001)

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
vector_field_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        typ='se',
                        sigma_f=1.)

vector_field_GP = train_gp(d.evecs_Lc.reshape(d.n, -1)[train_ind], 
                                  d.vectors[train_ind],
                                   dim=d.vertices.shape[1],
                                  epochs=1000,
                                   # n_inducing_points=20,
                                    # kernel=vector_field_kernel,
                                  noise_variance=0.001)

# =============================================================================
# Make new predictions
# =============================================================================
x_test = d.evecs_L.reshape(-1, n_eigenpairs)
# x_test = d.evecs_Lc.reshape(d.n, -1)
y_pred_mean, _ = manifold_GP.predict_f(x_test)

x_test = d.evecs_Lc.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(x_test)
f_pred_mean = f_pred_mean.numpy().reshape(d.n,-1)

# =============================================================================
# Plot
# =============================================================================
ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", X, faces)
ps_cloud = ps.register_point_cloud("Training points", d.vertices[train_ind])
ps_cloud.add_vector_quantity("Training vectors", d.vectors[train_ind], color=(0., 0., 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", y_pred_mean[test_ind])
ps_cloud.add_vector_quantity("Predicted vectors", f_pred_mean[test_ind], color=(1., 0., 0.), enabled=True)
ps.show()

# from misc import load_mesh
# from RVGP.geometry import sample_from_neighbourhoods
# from RVGP.kernels import ManifoldKernel
# from RVGP import data, train_gp
# import polyscope as ps


# # Load mesh
# vertices, faces = load_mesh('sphere')
# dim_emb = vertices.shape[1]

# # Form data object
# d = data(vertices, faces, n_eigenpairs=3)

# d.random_vector_field()
# d.smooth_vector_field(t=100)

# manifold_kernel = ManifoldKernel((d.evecs_L, d.evals_L), 
#                         nu=3/2, 
#                         kappa=5, 
#                         sigma_f=1)

# sp_to_manifold_gp = train_gp(d.evecs_L,
#                              d.vertices,
#                              # n_inducing_points=20,
#                              kernel=manifold_kernel,
#                              epochs=1000,
#                              noise_variance=0.001)

# vector_field_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
#                         nu=3/2, 
#                         kappa=5, 
#                         sigma_f=1)

# sp_to_vector_field_gp = train_gp(d.evecs_Lc.reshape(d.n, -1), 
#                                  d.vectors,
#                                  dim=d.vertices.shape[1],
#                                  epochs=1000,
#                                  # n_inducing_points=20,
#                                  kernel=vector_field_kernel,
#                                  noise_variance=0.001)

# # n_test = 2
# # test_points = sample_from_neighbourhoods(d.evecs_Lc.reshape(d.n, -1), k=2, n=n_test)
# test_points = d.evecs_L.reshape(d.n, -1)
# manifold_pred_mean, _ = sp_to_manifold_gp.predict_f(test_points)

# test_points = d.evecs_Lc.reshape(d.n, -1)
# vector_field_pred_mean, _ = sp_to_vector_field_gp.predict_f(test_points.reshape(len(test_points)*d.vertices.shape[1], -1))
# vector_field_pred_mean = vector_field_pred_mean.numpy().reshape(len(test_points), -1)

# ps.init()
# ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
# ps_cloud = ps.register_point_cloud("Training points", d.vertices)
# ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0., 0., 1.), enabled=True)
# ps_cloud = ps.register_point_cloud("Predicted points", manifold_pred_mean)
# ps_cloud.add_vector_quantity("Predicted vectors", vector_field_pred_mean, color=(1., 0., 0.), enabled=True)
# ps.show()
