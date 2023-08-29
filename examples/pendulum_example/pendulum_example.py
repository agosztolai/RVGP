#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from misc import simulate_pendulum, cylinder_m_to_3d, plot_vector_field, mesh_to_polyscope, phase_to_coordinates

from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel
from RVGP.geometry import furthest_point_sampling
import polyscope as ps

from RVGP import data, train_gp

# =============================================================================
# Generate data
# =============================================================================
n_points=20
grid, vector_field, rollouts = simulate_pendulum(n_points=n_points, n_traj=10)

# =============================================================================
# Phase plot
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
for i in range(rollouts.shape[0]):
    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.gca().set_aspect("equal")

# =============================================================================
# Convert from phase to R^3
# =============================================================================

xyz = phase_to_coordinates(rollouts[:,:,0], radius=3)
rollouts = np.dstack([xyz , rollouts[:,:,1].reshape(rollouts.shape[0],-1,1)])

# =============================================================================
# Create data object
# =============================================================================
X = rollouts[:,:-1,:]
X = X.reshape(-1, X.shape[2])

# get vectors
f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.reshape(-1, f.shape[2])

#uniform sampling of the datapoints
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.03) 
X = X[sample_ind]
f = f[sample_ind]

d = data(X, vectors=f, dim_man=2, n_eigenpairs=None, n_neighbors=20,)

# =============================================================================
# Train GPs
# =============================================================================
# spectrum to manifold
sp_to_manifold_gp = train_gp(d.evecs_Lc.reshape(d.n, -1),
                             d.vertices,
                             dim=1,
                             epochs=2000,
                             noise_variance=0.001)


# spectrum to vector field
# kernel2 = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
#                         nu=3/2, 
#                         kappa=5, 
#                         sigma_f=1)

# sp_to_vector_field_gp = train_gp(d.evecs_Lc.reshape(d.n, -1), 
#                                   d.vectors,
#                                   dim=d.vertices.shape[1],
#                                   epochs=2000,
#                                   kernel=kernel2,
#                                   noise_variance=0.001)

# # manifold to spectrum
# manifold_to_sp_gp = train_gp(d.vertices,
#                               d.evecs_Lc.reshape(d.n, -1),
#                               dim=1,
#                               epochs=2000,
#                               noise_variance=0.01)


# =============================================================================
# Plot predictions
# =============================================================================

test_points = d.evecs_Lc.reshape(d.n, -1)
X_pred, _ = sp_to_manifold_gp.predict_f(test_points)

# f_pred, _ = sp_to_vector_field_gp.predict_f(test_points.reshape(len(test_points)*d.vertices.shape[1], -1))
# f_pred = f_pred.numpy().reshape(len(test_points), -1)

# ax = graph(d.G)
# ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=2)
# ax.quiver(X_pred[:,0], X_pred[:,1], X_pred[:,2], f_pred[:,0], f_pred[:,1], f_pred[:,2], color='r', length=2)


mesh = cylinder_m_to_3d(grid, radius=3).reshape(n_points,n_points,3)

ps.init()
ps.set_up_dir("z_up")
klein_mesh = ps.register_surface_mesh(
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
ps_cloud = ps.register_point_cloud("Training points", d.vertices)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0., 0., 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", X_pred)
# ps_cloud.add_vector_quantity("Predicted vectors", f_pred, color=(1., 0., 0.), enabled=True)
ps.show()





# #%% project down onto plane and visualise
# def project_vectors_and_positions_onto_cylinder(positions, vectors):
#     n = positions.shape[0]
#     projected_vectors = np.zeros((n, 2))
#     projected_positions = np.zeros((n, 2))
    
#     for i in range(n):
#         x, y, z = positions[i]
#         vector = vectors[i]

#         # Compute the angular coordinate theta
#         theta = np.arctan2(y, x)
        
#         if theta < 0:
#             theta += 2 * np.pi

#         # Define the gauge directions
#         gauge_theta = np.array([-y, x, 0])
#         gauge_z = np.array([0, 0, 1])

#         # Normalize the gauge direction in the theta direction
#         gauge_theta /= np.linalg.norm(gauge_theta)

#         # Define the gauge matrix
#         gauge = np.column_stack((gauge_theta, gauge_z))

#         # Project the vector onto the gauge directions
#         projected_vector = np.dot(gauge.T, vector)

#         projected_vectors[i] = projected_vector
#         projected_positions[i] = [theta, z]

#     return projected_positions, projected_vectors


# X_proj, f_proj = project_vectors_and_positions_onto_cylinder(d.vertices, d.vectors)
# X_pred_proj, f_pred_proj = project_vectors_and_positions_onto_cylinder(X_pred, f_pred)


# scale = 100
# width = 0.004
# fig, ax = plt.subplots(1, 1, figsize=(2, 2))
# ax.quiver(X_proj[:,0], X_proj[:,1], f_proj[:,0], f_proj[:,1],color='b')
# ax.quiver(X_pred_proj[:,0], X_pred_proj[:,1], f_pred_proj[:,0], f_pred_proj[:,1], color='r')
# #for i in range(rollouts.shape[0]):
# #    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
# plt.xlabel("Position")
# plt.ylabel("Momentum")
# plt.gca().set_aspect("equal")

# #%% test the predictions of manifold to spectrum conversion


# sp_pred, _ = manifold_to_sp_gp.predict_f(d.vertices)
# X_pred, _ = sp_to_manifold_gp.predict_f(sp_pred)
# f_pred, _ = sp_to_vector_field_gp.predict_f(test_points.reshape(len(d.vertices)*d.vertices.shape[1], -1))
# f_pred = f_pred.numpy().reshape(len(d.vertices), -1)
# #f_pred[:, 0] = np.mod(f_pred[:, 0] + np.pi, 2 * np.pi) - np.pi

# # l2_error = np.linalg.norm(f.ravel() - f_pred.ravel()) / len(f.ravel())
# # print("Relative l2 error is {}".format(l2_error))

# ax = graph(d.G)
# ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=2)
# ax.quiver(X_pred[:,0], X_pred[:,1], X_pred[:,2], f_pred[:,0], f_pred[:,1], f_pred[:,2], color='r', length=2)


# X_proj, f_proj = project_vectors_and_positions_onto_cylinder(X_pred, f_pred)
# scale = 100
# width = 0.004
# fig, ax = plt.subplots(1, 1, figsize=(2, 2))
# ax.quiver(X_proj[:,0], X_proj[:,1], f_proj[:,0], f_proj[:,1],)
# #for i in range(rollouts.shape[0]):
# #    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
# plt.xlabel("Position")
# plt.ylabel("Momentum")
# plt.gca().set_aspect("equal")


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


# #%% create trjaectory


# len_t = 100
# h = 2 # step size

# # choose a point y to find x 
# idx = 0#90
# y0 = d.vertices[idx, :] # + np.array([0.1,-0.3])

# trajectory = [y0]
# vectors = []


# # looping over length of trajectory
# for i in range(len_t):
    
#     y = trajectory[-1]  
    
#     # predict the tangent space eigenvectors TU (logarithmic map)
#     x_, x_var = manifold_to_sp_gp.predict_y(y.reshape(-1,1).T)
    
#     # predict the manifold point from TU (exponential map)
#     y_pred, _ = sp_to_manifold_gp.predict_f(x_) # should be similar to y
#     y_pred = y_pred.numpy()
    
#     # predict the vector from the TU
#     x_ = x_.numpy()
#     f_pred_, f_var = sp_to_vector_field_gp.predict_f(x_.reshape(1*d.vertices.shape[1], -1))
#     f_pred_ = f_pred_.numpy().reshape(1, -1)
   
#     # perform euler iteration    
#     y1 = y + h * f_pred_    
    
#     # append to trajectory
#     trajectory.append(y1.squeeze())
#     vectors.append(f_pred_)


# # create plot 3d
# ax = graph(d.G)
# #ax.quiver(X[:,0], X[:,1], f[:,0], f[:,1], color='g', scale=0.3)
# ax.quiver(X[:,0], X[:,1],X[:,2],  f[:,0], f[:,1], f[:,2],  color='g', length=1)

# for i in range(0,len_t,3):
#     y = trajectory[i]
#     f_traj = vectors[i]
#     #ax.quiver(y[0], y[1],  f_traj[0,0], f_traj[0,1], color='r', scale=0.3)
#     ax.quiver(y[0], y[1], y[2], f_traj[0,0], f_traj[0,1],  f_traj[0,2], color='r', length=5)

# # create plot 2d
# X_proj, f_proj = project_vectors_and_positions_onto_cylinder(X_pred, f_pred)
# scale = 100
# width = 0.004
# fig, ax = plt.subplots(1, 1, figsize=(2, 2))
# ax.quiver(X_proj[:,0], X_proj[:,1], f_proj[:,0], f_proj[:,1],)
# #for i in range(rollouts.shape[0]):
# #    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
# plt.xlabel("Position")
# plt.ylabel("Momentum")
# plt.gca().set_aspect("equal")

# X_proj_traj, f_proj_traj = project_vectors_and_positions_onto_cylinder(np.vstack(trajectory)[:-1,:], np.vstack(vectors))
# X_proj_traj[:, 0] = np.mod(X_proj_traj[:, 0] , 2 * np.pi) 

# for i in range(0,len_t,3):
#     y = X_proj_traj[i]
#     f_traj =  f_proj_traj[i]
#     #ax.quiver(y[0], y[1],  f_traj[0,0], f_traj[0,1], color='r', scale=0.3)
#     ax.quiver(y[0], y[1], f_traj[0], f_traj[1],  color='r')

