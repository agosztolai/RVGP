#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from misc import (simulate_pendulum, 
                  phase_to_coordinates, 
                  plot_vector_field, 
                  cylinder_m_to_3d, 
                  mesh_to_polyscope,
                  integrate_trajectory,
                  project_vectors_and_positions_onto_cylinder,
                  plot_wrapped_data)
from RVGP import data, train_gp
from RVGP.kernels import ManifoldKernel
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import polyscope as ps
import random


random.seed(10)

# =============================================================================
# Generate data
# =============================================================================
n_points=60 #number of gridpoints
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

d = data(X, vectors=f, dim_man=2, n_eigenpairs=n_eigenpairs, n_neighbors=10,)


# =============================================================================
# Lc to manifold
# =============================================================================

positional_encoding = d.evecs_Lc.reshape(d.n, -1)
kernel = ManifoldKernel((d.evecs_Lc.reshape(d.n, -1), np.tile(d.evals_Lc,3)), 
                          nu=3/2, 
                          kappa=5, 
                          sigma_f=1)



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
# Predict with GP
# =============================================================================

x_test = positional_encoding
y_pred, _ = manifold_GP.predict_f(x_test)
f_pred, _ = vector_GP.predict_f(x_test.reshape(d.n*3, -1))
f_pred = f_pred.numpy().reshape(d.n, -1)

#%%
# =============================================================================
# Simulate trajectory
# =============================================================================

len_t = 300 # length of trajectory
h = 5 # step size 

# initial starting positions of trajectory on manifold
n = 10
initial_states = np.stack(
    [np.array([(m + 1) * 2 * np.pi / (n + 1) - 0.01]) for m in range(n)],
    axis=0,
)

# convert intiail positions onto cylinder
initial_states = np.hstack([radius*np.cos(initial_states),radius*np.sin(initial_states), np.zeros([n,1])])

# integrate trajectory using Euler method
trajectories = [];vectors=[]
for i in range(n):    
    traj, vec = integrate_trajectory(initial_states[i,:], d, manifold_GP, vector_GP, h=h, len_t=len_t)
    trajectories.append(traj)
    vectors.append(vec)



# =============================================================================
# Plotting 3D
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

color = iter(cm.rainbow(np.linspace(0, 1, n)))
for i in range(n):
    c = next(color)
    ps.register_point_cloud("Trajectory {}".format(i), trajectories[i], radius=0.01, color=c) #, radius=0.01)

ps.show()



# =============================================================================
# Plotting 2D
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_vector_field(grid, vector_field, color="black", ax=ax, scale=100, width=0.004)
color = iter(cm.rainbow(np.linspace(0, 1, n)))
for i in range(n):
    c = next(color)
    trajectory, _ = project_vectors_and_positions_onto_cylinder(trajectories[i], vectors[i])
    plot_wrapped_data(trajectory[:, 0], trajectory[:, 1], c)

plt.savefig('pendulum_example_cylinder_trajectories.svg')






