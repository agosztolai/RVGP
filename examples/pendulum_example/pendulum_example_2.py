#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from misc import (simulate_pendulum, 
                  phase_to_coordinates, 
                  plot_vector_field, 
                  cylinder_m_to_3d, 
                  mesh_to_polyscope)
from RVGP.geometry import furthest_point_sampling
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import gpflow
import matplotlib.pyplot as plt
import polyscope as ps

# =============================================================================
# Generate data
# =============================================================================
n_points=20 #number of gridpoints
n_traj=20 #number of simulated trajectories
n_eigenpairs=100
radius=3

grid, vector_field, rollouts = simulate_pendulum(n_points=n_points, n_traj=n_traj)

# =============================================================================
# Phase plot
# =============================================================================
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
# for i in range(rollouts.shape[0]):
#     plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
# plt.gca().set_aspect("equal")

# =============================================================================
# Convert from phase to R^3
# =============================================================================
xyz = phase_to_coordinates(rollouts[:,:,0], radius=radius)
rollouts = np.dstack([xyz , rollouts[:,:,1].reshape(rollouts.shape[0],-1,1)])

# =============================================================================
# Create data object
# =============================================================================
X = rollouts[:,:-1,:]
X = X.reshape(-1, X.shape[2])

sf = 100

X *= (1.5+X[:,[2]]**2) / sf

# get vectors
f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.reshape(-1, f.shape[2])

#uniform sampling of the datapoints
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.02) 
X = X[sample_ind]
f = f[sample_ind]

d = data(X, vectors=f, dim_man=2, n_eigenpairs=n_eigenpairs, n_neighbors=20,)


#%%

# =============================================================================
# Lc to manifold
# =============================================================================

positional_encoding = d.evecs_Lc.reshape(d.n, -1)
kernel = gpflow.kernels.RBF()
manifold_GP = train_gp(positional_encoding,
                       d.vertices,
                       kernel=kernel,
                       dim=1,
                       noise_variance=0.001)



# =============================================================================
# Lc to vectors
# =============================================================================

kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                         nu=3/2, 
                         kappa=5, 
                         sigma_f=1)

vector_GP = train_gp(positional_encoding,
                       d.vectors,
                       kernel=kernel,
                       dim=3,
                       noise_variance=0.001)

# =============================================================================
# manifold to Lc
# =============================================================================

kernel = gpflow.kernels.Matern52()
spectral_GP = train_gp(d.vertices,
                       positional_encoding,
                       kernel=kernel,
                       dim=1,
                       lr=1,
                       noise_variance=0.001)


#%%
# =============================================================================
# Predict with GP
# =============================================================================

#x_test = positional_encoding
x_test, _ = spectral_GP.predict_f(d.vertices)
x_test = x_test.numpy()
y_pred, _ = manifold_GP.predict_f(x_test)
f_pred, _ = vector_GP.predict_f(x_test.reshape(d.n*3, -1))
f_pred = f_pred.numpy().reshape(d.n, -1)

#%%
# =============================================================================
# Simulate trajectory
# =============================================================================
len_t = 50
h = 2 #step size
idx = 0 #starting point

# looping over length of trajectory
y0 = d.vertices[idx, :] # + np.array([0.1,-0.3])
x0 = d.evecs_Lc.reshape(d.n,-1)[idx]

trajectory = [y0]
vectors = []
for i in range(len_t):
    
    y = trajectory[-1]  
    
    x_test, _ = spectral_GP.predict_f(y.reshape(-1,1).T)

        
    # predict the manifold point from TU (exponential map)
    y_pred_, _ = manifold_GP.predict_f(x_test) # should be similar to y
    y_pred_ = y_pred_.numpy()
    
    # predict the vector from the TU
    f_pred_, f_var = vector_GP.predict_f(x_test.numpy().reshape(3,-1))
    f_pred_ = f_pred_.numpy().reshape(1, -1)
   
    # perform euler iteration    
    y1 = y + h * f_pred_    
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    vectors.append(f_pred_)

trajectory = np.vstack(trajectory)[:-1,:]
vectors = np.vstack(vectors)

#%%
# =============================================================================
# Plotting
# =============================================================================
mesh = cylinder_m_to_3d(grid, radius=radius).reshape(n_points,n_points,3)

mesh *= (1+mesh[:,:,[2]]**2)

ps.init()
ps.set_up_dir("z_up")
ps.register_surface_mesh(
    f"cylinder",
    *mesh_to_polyscope(
            mesh,
            wrap_x=False,
            wrap_y=False,
            reverse_x=False,
        ),
    color=(39/255,119/255,177/255),
    smooth_shade=True,
    material="wax",
    enabled=False,
)
# ps.set_ground_plane_mode('none')
ps_cloud = ps.register_point_cloud("Training points", d.vertices*sf)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0., 0., 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", y_pred*sf, color=(1., 0., 0.),)
ps_cloud.add_vector_quantity("Predicted vectors", f_pred, color=(1., 0., 0.), enabled=True)
ps_cloud = ps.register_point_cloud("Trajectory", trajectory*sf,color=(0., 1., 0.),)
ps_cloud.add_vector_quantity("Trajectory tangent vectors", vectors, color=(0., 1., 0.), enabled=True)
ps.show()











