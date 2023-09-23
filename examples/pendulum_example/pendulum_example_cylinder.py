#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from misc import (simulate_pendulum, 
                  phase_to_coordinates, 
                  plot_vector_field, 
                  cylinder_m_to_3d, 
                  mesh_to_polyscope)
from RVGP.geometry import furthest_point_sampling, closest_manifold_point
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import gpflow
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import polyscope as ps
import random


random.seed(10)

# =============================================================================
# Generate data
# =============================================================================
n_points=40 #number of gridpoints
n_traj=2 #number of simulated trajectories
n_eigenpairs=300 # 300
steps=2
radius=2

grid, vector_field, rollouts = simulate_pendulum(n_points=n_points, n_traj=n_traj, steps=steps)

# =============================================================================
# Phase plot
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
plt.savefig('pendulum_true_dynamics.svg')


# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
# for i in range(rollouts.shape[0]):
#     plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=2)
# plt.savefig('pendulum_sampled_trajectories.svg')
# #plt.gca().set_aspect("equal")

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

# get vectors
f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.reshape(-1, f.shape[2])

#uniform sampling of the datapoints
#sample_ind, _ = furthest_point_sampling(X,stop_crit=0.01) 
#X = X[sample_ind]
#f = f[sample_ind]


d = data(X, vectors=f, dim_man=2, n_eigenpairs=n_eigenpairs, n_neighbors=10,)

# =============================================================================
# Training data plot
# =============================================================================

#fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
#sub_data = rollouts.reshape(10*2001,-1)[sample_ind]
#plt.scatter(sub_data[:, 0], sub_data[:, 1], s=20)
#plt.savefig('pendulum_subsampled_training_data.svg')

#fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#plt.scatter(rollouts.reshape)


#from RVGP.utils import construct_embedding

#D = 10; k = 5; epsilon = 0.4
#evecs, evals = construct_embedding(d.evecs_Lc, d.evals_Lc, D, k, epsilon)
#d.evecs_Lc = evecs
#d.evals_Lc = evals

#%%

# =============================================================================
# Lc to manifold
# =============================================================================

positional_encoding = d.evecs_Lc.reshape(d.n, -1)
# kernel = gpflow.kernels.RBF()
# manifold_GP = train_gp(positional_encoding,
#                        d.vertices,
#                        kernel=kernel,
#                        dim=1,
#                        noise_variance=0.001)

kernel = ManifoldKernel((d.evecs_Lc.reshape(d.n, -1), np.tile(d.evals_Lc,3)), 
                          nu=3/2, 
                          kappa=5, 
                          sigma_f=1)

#gpflow.set_trainable(kernel.nu, False)
#gpflow.set_trainable(kernel.kappa, False)

manifold_GP = train_gp(positional_encoding,
                        d.vertices,
                        kernel=kernel,
                        n_inducing_points=100,
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
                       n_inducing_points=100,
                       noise_variance=0.001)

# =============================================================================
# manifold to Lc
# =============================================================================

# kernel = gpflow.kernels.Matern52()
# spectral_GP = train_gp(d.vertices,
#                        positional_encoding,
#                        kernel=kernel,
#                        dim=1,
#                        lr=1,
#                        noise_variance=0.001)


#%%
# =============================================================================
# Predict with GP
# =============================================================================

x_test = positional_encoding
#x_test, _ = spectral_GP.predict_f(d.vertices)
#x_test = x_test.numpy()
y_pred, _ = manifold_GP.predict_f(x_test)
f_pred, _ = vector_GP.predict_f(x_test.reshape(d.n*3, -1))
f_pred = f_pred.numpy().reshape(d.n, -1)

#%%
# =============================================================================
# Simulate trajectory
# =============================================================================
len_t = 300
h = 5 #5 #step size

# looping over length of trajectory

#dx = 0.5
#y0 = np.array([np.cos(np.pi+dx), np.sin(np.pi+dx), 0])
#y0[:2] = y0[:2] + 0.1*np.cos(8 * y0[2]) 

def integrate_trajectory(y0):
    
    trajectory = [y0]
    vectors = []
    
    for i in range(len_t):
        
        y = trajectory[-1]  
        
        #x_test, _ = spectral_GP.predict_f(y.reshape(-1,1).T)    
        # predict the eigenvectors: TU
        _, x_ = closest_manifold_point(y.reshape(-1,1).T, d, nn=10)
            
        # predict the manifold point from TU (exponential map)
        y_pred_, _ = manifold_GP.predict_f(x_) # should be similar to y
        y_pred_ = y_pred_.numpy()
        
        # predict the vector from the TU
        f_pred_, f_var = vector_GP.predict_f(x_.reshape(3,-1))
        f_pred_ = f_pred_.numpy().T
       
        # perform euler iteration    
        y1 = y_pred_ + h * f_pred_    
        
        # append to trajectory
        trajectory.append(y1.squeeze())
        vectors.append(f_pred_)
    
    trajectory = np.vstack(trajectory)[:-1,:]
    vectors = np.vstack(vectors)
    
    return trajectory, vectors

#y0 = d.vertices[29, :] 
#trajectory_1, vectors_1 = integrate_trajectory(y0)

#y0 = d.vertices[23, :] 
#trajectory_2, vectors_2 = integrate_trajectory(y0)


n = 10
initial_states = np.stack(
    [np.array([(m + 1) * 2 * np.pi / (n + 1) - 0.01]) for m in range(n)],
    axis=0,
)

initial_states = np.hstack([radius*np.cos(initial_states),radius*np.sin(initial_states), np.zeros([n,1])])

trajectories = [];vectors=[]
for i in range(n):    
    traj, vec = integrate_trajectory(initial_states[i,:])
    trajectories.append(traj)
    vectors.append(vec)




# =============================================================================
#%%
# =============================================================================
# Plotting
# =============================================================================
mesh = cylinder_m_to_3d(grid, radius=radius)

mesh = mesh.reshape(n_points,n_points,3)

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
    color=(0.612,0.612,0.612),
    smooth_shade=True,
    material="wax",
    enabled=False,
)
ps.set_ground_plane_mode('none')
ps_cloud_training = ps.register_point_cloud("Training points", d.vertices, color=(0.132, 0.132, 0.132))
ps_cloud_training.add_vector_quantity("Training vectors", d.vectors, color=(0.132, 0.132, 0.132), enabled=True)

ps_cloud = ps.register_point_cloud("Predicted points", y_pred, color=(1., 0., 0.),)
ps_cloud.add_vector_quantity("Predicted vectors", f_pred, color=(1., 0., 0.), enabled=True)

#ps_cloud_trajectory_1 = ps.register_point_cloud("Trajectory 1", trajectory_1,color=(1., 0.1171, 0.), radius=0.01)
#ps_cloud_trajectory_1.add_vector_quantity("Trajectory tangent vectors 1", vectors_1, color=(1., 0.171, 0.),radius=0.005, length=0.1, enabled=True)

#ps_cloud_trajectory_2 = ps.register_point_cloud("Trajectory 2", trajectory_2,color=(1., 0.619, 0.), radius=0.01)
#ps_cloud_trajectory_2.add_vector_quantity("Trajectory tangent vectors 2", vectors_2, color=(1., 0.619, 0.), radius=0.005, length=0.1, enabled=True)

color = iter(cm.rainbow(np.linspace(0, 1, n)))

for i in range(n):
    c = next(color)
    ps.register_point_cloud("Trajectory {}".format(i), trajectories[i], radius=0.01, color=c) #, radius=0.01)

ps.show()



#%%
from misc import project_vectors_and_positions_onto_cylinder


def plot_wrapped_data(x, y, c):
    # Sort the data by x-values

    # Initialize lists to store continuous segments
    segments_x = [x[0]]
    segments_y = [y[0]]

    # Loop through the sorted points to find discontinuities
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        
        # Check for discontinuity (wrap-around)
        if np.abs(dx) > np.pi:
            # Break the line at this point
            plt.plot(segments_x, segments_y, marker='o',c=c, linewidth=2)
            segments_x = [x[i]]
            segments_y = [y[i]]
        else:
            segments_x.append(x[i])
            segments_y.append(y[i])
            
    # Plot the last segment
    plt.plot(segments_x, segments_y, marker='o',c=c, linewidth=2)


# trajectory_1_ = trajectory_1.copy()
# trajectory_1_[:,:2] = trajectory_1_[:,:2] + 0.1 * np.sin(4 * trajectory_1_[:,[2]])
# trajectory_1_, _ = project_vectors_and_positions_onto_cylinder(trajectory_1_, vectors_1)

# trajectory_2_ = trajectory_2.copy()
# trajectory_2_[:,:2] = trajectory_2_[:,:2] + 0.1 * np.sin(4 * trajectory_2_[:,[2]])
# trajectory_2_, _ = project_vectors_and_positions_onto_cylinder(trajectory_2_, vectors_2)

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
# plot_wrapped_data(trajectory_1_[:, 0], trajectory_1_[:, 1], (1., 0.1171, 0.))
# plot_wrapped_data(trajectory_2_[:, 0], trajectory_2_[:, 1], (1., 0.619, 0.))

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
color = iter(cm.rainbow(np.linspace(0, 1, n)))
for i in range(n):
    c = next(color)
    trajectory, _ = project_vectors_and_positions_onto_cylinder(trajectories[i], vectors[i])
    plot_wrapped_data(trajectory[:, 0], trajectory[:, 1], c)

#plt.savefig('pendulum_example_cylinder_trajectories.svg')






