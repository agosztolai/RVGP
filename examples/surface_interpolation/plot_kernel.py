#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import RVGP
from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
from RVGP.kernels import ManifoldKernel
import polyscope as ps
import numpy as np
import gpflow

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=100
vertices, faces = load_mesh('sphere') #see /examples/data for more objects

# =============================================================================
# Subsample 
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, spacing=0.05)
X = vertices[sample_ind]

train_ind =  np.random.choice(np.arange(len(X)), size=int(0.5*len(X)))
test_ind = [i for i in range(len(X)) if i not in train_ind]

# =============================================================================
# Create data object
# =============================================================================
d = RVGP.create_data_object(X, faces, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)

# =============================================================================
# Define GP for vector field over manifold
# =============================================================================
vector_field_kernel = ManifoldKernel(d, 
                                     nu=3/2, 
                                     kappa=3, 
                                     typ='matern',
                                     sigma_f=1.)

vector_field_GP = gpflow.models.GPR((d.evecs_Lc.reshape(d.n*vertices.shape[1], -1), 
                                     d.vectors.reshape(d.n*vertices.shape[1], -1)), 
                        kernel=vector_field_kernel, 
                        noise_variance=0.001,
                        )

# =============================================================================
# Predict with GPs
# =============================================================================
# test_x = d.evecs_Lc
# f_pred_mean, _ = vector_field_GP.predict_f(test_x)
# f_pred_mean = f_pred_mean.numpy().reshape(d.n, -1)

# =============================================================================
# Plot kernel
# =============================================================================
K = vector_field_kernel(d.evecs_Lc)
n, dim = X.shape
K = K.numpy().reshape(n,dim,n,dim).swapaxes(1,2).swapaxes(2,3)
K = K[100]

# for i, K_ in enumerate(K):
#     u, v = np.linalg.eig(K_)
#     idx = np.abs(u).argsort()[::-1]   
#     u = u[idx]
#     v = v[:,idx]
#     if np.iscomplex(u).any():
#         K[i] = 0
#     else:
#         K[i] = u/u.max()*v

ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)

ps_cloud = ps.register_point_cloud("Reference point", X[[100]])

# ps_cloud = ps.register_point_cloud("Training points", X)
# ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0, 0, 0), enabled=True)

ps_cloud_1 = ps.register_point_cloud("Points 1", X)
ps_cloud_1.add_vector_quantity("Kernel 1", K[:,:,0], color=(0, 0, 256), enabled=True)

ps_cloud_2 = ps.register_point_cloud("Points 2", X)
ps_cloud_2.add_vector_quantity("TKernel 2", K[:,:,1], color=(0, 256, 0), enabled=True)

ps_cloud_3 = ps.register_point_cloud("Points 3", X)
ps_cloud_3.add_vector_quantity("TKernel 3", K[:,:,2], color=(258, 0, 0), enabled=True)
ps.show()
