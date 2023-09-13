#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from misc import simulate_pendulum, cylinder_m_to_3d, plot_vector_field, mesh_to_polyscope, phase_to_coordinates, project_vectors_and_positions_onto_cylinder

from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel
from RVGP.geometry import furthest_point_sampling
import polyscope as ps
import gpflow

from RVGP import data, train_gp

# =============================================================================
# Generate data
# =============================================================================
n_points=20
grid, vector_field, rollouts = simulate_pendulum(n_points=n_points, n_traj=10, steps=2)

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
radius = 2
#xyz = phase_to_coordinates(rollouts[:,:,0], radius=1+ 0.2*np.sin(4*rollouts[:,:,1]))
xyz = phase_to_coordinates(rollouts[:,:,0], radius=radius + 0.2*rollouts[:,:,1]**2)
#xyz = phase_to_coordinates(rollouts[:,:,0], radius=radius)
rollouts = np.dstack([xyz , rollouts[:,:,1].reshape(rollouts.shape[0],-1,1)])

# =============================================================================
# Create data object
# =============================================================================
X = rollouts[:,:-1,:]*3 
X = X.reshape(-1, X.shape[2])

# get vectors
f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.reshape(-1, f.shape[2])

#uniform sampling of the datapoints
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.03) 
X = X[sample_ind]
f = f[sample_ind]

d = data(X, vectors=f, dim_man=2, n_eigenpairs=500, n_neighbors=10, n_geodesic_neighbors=5,)

#%%
# =============================================================================
# Train GPs
# =============================================================================

# connection laplacian spectrum to manifold
# manifold_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
#                                 nu=3/2, 
#                                 kappa=5, 
#                                 typ='matern',
#                                 sigma_f=1.)

# Lc_to_manifold_gp = train_gp(d.evecs_Lc.reshape(d.n, -1),
#                               d.vertices,
#                               dim=3,
#                               test_size=0.05,
#                               epochs=4000,
#                               kernel = manifold_kernel,
#                               noise_variance=0.001)

#%%

# graph laplacian spectrum to manifold
manifold_kernel = ManifoldKernel((d.evecs_L, d.evals_L), 
                        nu=3/2, 
                        kappa=5, 
                        typ='matern',
                        sigma_f=1.)

L_to_manifold_gp = train_gp(d.evecs_L.reshape(d.n, -1),
                             d.vertices,
                             dim=1,
                             test_size=0.05,
                             epochs=2000,
                             kernel = manifold_kernel,
                             noise_variance=0.001)


# spectrum to vector field
kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                         nu=3/2, 
                         kappa=5, 
                         sigma_f=1)

Lc_to_vector_field_gp = train_gp(d.evecs_Lc.reshape(d.n, -1), 
                                  d.vectors,
                                  dim=d.vertices.shape[1],
                                  epochs=2000,
                                  kernel=kernel,
                                  noise_variance=0.001)

# manifold to spectrum
kernel = gpflow.kernels.Matern32()
manifold_to_Lc_gp = train_gp(d.vertices,
                              d.evecs_Lc.reshape(d.n, -1),
                              dim=1,
                              epochs=2000,
                              kernel=kernel,
                              noise_variance=0.001)


kernel = gpflow.kernels.Matern32()
manifold_to_L_gp = train_gp(d.vertices,
                            d.evecs_L.reshape(d.n, -1),
                            dim=1,
                            epochs=2000,
                            kernel=kernel,
                            noise_variance=0.001)

#%%
# =============================================================================
# Predict all points
# =============================================================================
test_points = d.evecs_Lc.reshape(d.n, -1)

f_pred, _ = Lc_to_vector_field_gp.predict_f(test_points.reshape(len(test_points)*d.vertices.shape[1], -1))
f_pred = f_pred.numpy().reshape(len(test_points), -1)

#X_pred_Lc, _ = Lc_to_manifold_gp.predict_f(d.evecs_Lc.reshape(d.n, -1))

#X_pred_Lc, _ = Lc_to_manifold_gp.predict_f(d.evecs_Lc)
#X_pred_Lc = X_pred_Lc.numpy().reshape(d.n,-1)
#X_pred_proj, f_pred_proj = project_vectors_and_positions_onto_cylinder(X_pred_Lc, f_pred)


X_pred_L, _ = L_to_manifold_gp.predict_f(d.evecs_L.reshape(d.n, -1))
X_pred_proj, f_pred_proj = project_vectors_and_positions_onto_cylinder(X_pred_L, f_pred)


# # %% Plot the kernel


# # k = kernel.matrix(kernel_params, phase_space, phase_space)

# # i = int(n_points ** 2 / 2)
# # plt.contourf(
# #     phase_space[:, 0].reshape(n_points, n_points),
# #     phase_space[:, 1].reshape(n_points, n_points),
# #     k[:, i, 0, 0].reshape(n_points, n_points),
# #     50,
# # )
# # plt.scatter(phase_space[i, 0], phase_space[i, 1])
# # plt.colorbar()

#%%
# =============================================================================
# Simulate trajectory
# =============================================================================
len_t = 50
h = 2 #step size
idx = 0 #starting point

# looping over length of trajectory
y0 = d.vertices[idx, :] # + np.array([0.1,-0.3])
trajectory = [y0]
vectors = []
for i in range(len_t):
    
    y = trajectory[-1]  
    
    # predict the Lc eigenvectors
    x_Lc, _ = manifold_to_Lc_gp.predict_y(y.reshape(-1,1).T)
    
    # predict the L eigenvectors
    x_L, _ = manifold_to_L_gp.predict_y(y.reshape(-1,1).T)
    
    # predict the manifold point from Lc 
    #y_pred_Lc, _ = Lc_to_manifold_gp.predict_f(x_Lc.numpy().reshape(3,-1)) # should be similar to y
    #y_pred_Lc = y_pred_Lc.numpy().T

    # predict the manifold point from L
    y_pred_L, _ = L_to_manifold_gp.predict_f(x_L) # should be similar to y
    y_pred_L = y_pred_L.numpy()
    
    # predict the vector from the TU
    f_pred_, f_var = Lc_to_vector_field_gp.predict_f(x_Lc.numpy().reshape(1*d.vertices.shape[1], -1))
    f_pred_ = f_pred_.numpy().reshape(1, -1)
   
    # perform euler iteration    
    y1 = y_pred_L + h * f_pred_    
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    vectors.append(f_pred_)

trajectory = np.vstack(trajectory)[:-1,:]
vectors = np.vstack(vectors)

#%%
# =============================================================================
# Plot graph
# =============================================================================
# ax = graph(d.G)

# =============================================================================
# Plot in projected 2D view by matplotlib
# =============================================================================

from misc import plot_wrapped_data

X_proj, f_proj = project_vectors_and_positions_onto_cylinder(d.vertices, d.vectors)
X_proj_traj, f_proj_traj = project_vectors_and_positions_onto_cylinder(trajectory, vectors)
X_proj_traj[:, 0] = np.mod(X_proj_traj[:, 0] , 2 * np.pi) 

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
#ax.quiver(X_proj[:,0], X_proj[:,1], f_proj[:,0], f_proj[:,1], width=0.008, scale=100)
# for i in range(rollouts.shape[0]):
#     plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
# for i in range(rollouts.shape[0]):
#plt.scatter(X_proj_traj[:, 0], X_proj_traj[:, 1], s=3, c='r')
#ax.quiver(X_proj_traj[:,0], X_proj_traj[:,1], f_proj_traj[:,0], f_proj_traj[:,1], color='r', width=0.008, scale=1)
plot_wrapped_data(X_proj_traj[:,0], X_proj_traj[:,1])

#plt.scatter(X_proj_traj[:,0], X_proj_traj[:,1],c='red')
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")
#plt.savefig('sample_trajectory_pendulum.png')

#%%

# =============================================================================
# Plot in 3D by polyscope
# =============================================================================
mesh = cylinder_m_to_3d(grid, radius=radius).reshape(n_points,n_points,3)

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
    # color=(28/255,99/255,227/255),
    # color=(1, 1, 1),
    color=(39/255,119/255,177/255), # matplotlib blue
    # color=(252/255,128/255,43/255), # matplotlib orange
    # color=(51/255, 159/255, 54/255), # matplotlib green
    # color=(217/255, 95/255, 2/255), # colorbrewer orange
    # color=(231/255, 41/255, 139/255), # colorbrewer magenta
    smooth_shade=True,
    material="wax",
    enabled=False,
)
ps.set_ground_plane_mode('none')
ps_cloud = ps.register_point_cloud("Training points", d.vertices)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0., 0., 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", X_pred_L, color=(1., 0., 0.),)
ps_cloud.add_vector_quantity("Predicted vectors", f_pred, color=(1., 0., 0.), enabled=True)
#ps_cloud = ps.register_point_cloud("Trajectory", trajectory,color=(1., 0., 0.),)
#ps_cloud.add_vector_quantity("Trajectory tangent vectors", vectors, color=(1., 0., 0.), enabled=True)
ps.show()
