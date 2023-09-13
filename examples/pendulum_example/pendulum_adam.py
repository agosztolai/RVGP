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

X *= (1+X[:,[2]]**2)

# get vectors
f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.reshape(-1, f.shape[2])

#uniform sampling of the datapoints
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.02) 
X = X[sample_ind]
f = f[sample_ind]

d = data(X, vectors=f, dim_man=2, n_eigenpairs=n_eigenpairs, n_neighbors=20,)

# =============================================================================
# Train GP
# =============================================================================
positional_encoding = d.evecs_Lc.reshape(d.n, -1)

manifold_GP = train_gp(positional_encoding,
                       X,
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
ps_cloud = ps.register_point_cloud("Training points", d.vertices)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0., 0., 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", y_pred_train, color=(1., 0., 0.),)
# ps_cloud.add_vector_quantity("Predicted vectors", f_pred, color=(1., 0., 0.), enabled=True)
#ps_cloud = ps.register_point_cloud("Trajectory", trajectory,color=(1., 0., 0.),)
#ps_cloud.add_vector_quantity("Trajectory tangent vectors", vectors, color=(1., 0., 0.), enabled=True)
ps.show()