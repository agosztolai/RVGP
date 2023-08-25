#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gpflow
from misc import load_mesh
from RVGP.kernels import ManifoldKernel
from RVGP import data, train_gp
import polyscope as ps

# =============================================================================
# Define manifold points
# =============================================================================
vertices, faces = load_mesh('sphere_fine')
dim_emb = vertices.shape[1]

# Form data object
d = data(vertices, faces, n_eigenpairs=50)

d.random_vector_field()
d.smooth_vector_field(t=100)

# =============================================================================
# Custom kernel
# =============================================================================
kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)

# =============================================================================
# Train GP for manifold
# =============================================================================
X = d.vertices
manifold_GP = train_gp(d.evecs_Lc.reshape(d.n, -1),
                              d.vertices,
                              # n_inducing_points=20,
                              epochs=1000,
                              noise_variance=0.001)

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
vector_field_GP = train_gp(d.evecs_Lc.reshape(d.n, -1), 
                                  d.vectors,
                                  dim=d.vertices.shape[1],
                                  epochs=1000,
                                  # n_inducing_points=20,
                                  kernel=kernel,
                                  noise_variance=0.001)

# =============================================================================
# Train GP for predicting spectral coordinates given some real point coordinates
# =============================================================================
manifold_GP_inv = gpflow.models.GPR((d.vertices, d.evecs_Lc.reshape(d.n, -1)),       
                                    kernel=gpflow.kernels.RBF(), noise_variance=0.01)

from RVGP.train_gp import optimize_model_with_scipy
manifold_GP_inv = optimize_model_with_scipy(manifold_GP_inv, 1000)

# manifold_GP_inv = train_gp(d.vertices,
#                               d.evecs_Lc.reshape(d.n, -1),
#                               # n_inducing_points=20,
#                               epochs=1000,
#                               noise_variance=0.01)


# =============================================================================
# Find trajectory in the vector field
# =============================================================================

# parameters
len_t = 100
h = 0.1 # step size
t0 = 100

y0 = d.vertices[t0, :]
trajectory = [y0]
trajectory_vectors = []

# looping over length of trajectory
for i in range(len_t):
    
    y = trajectory[-1]  
    
    # predict the eigenvectors: TU
    x_, x_var = manifold_GP_inv.predict_y(y.reshape(-1,1).T)
    
    # predict the manifold point from TU
    y_pred, _ = manifold_GP.predict_f(x_) # should be similar to y
    y_pred = y_pred.numpy()
    
    # predict the vector from the TU
    x_ = x_.numpy().reshape(-1,d.evecs_Lc.shape[-1])
    f_pred, f_var = vector_field_GP.predict_f(x_)
    f_pred = f_pred.numpy().reshape(-1,dim_emb)
   
    # perform euler iteration    
    y1 = y_pred + h * f_pred
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    trajectory_vectors.append(f_pred)

trajectory = np.vstack(trajectory)[1:]
trajectory_vectors = np.vstack(trajectory_vectors)

# create plot
ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", d.vertices)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0.0, 0.0, 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", trajectory)
ps_cloud.add_vector_quantity("Predicted vectors", trajectory_vectors, color=(1., 0., 0.), enabled=True)
ps.show()

