#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from dynamical_systems import FrictionPendulumSystem
from RVGP.geometry import closest_manifold_point


def plot_vector_field(X, 
                      Y, 
                      ax=None, 
                      color=None, 
                      scale=15, 
                      width=None, 
                      label=None, 
                      zorder=1
                      ):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    if color is None:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            np.hypot(Y[:, 0], Y[:, 1]),
            scale=scale,
            width=width,
            label=label,
            zorder=zorder,
            pivot="mid",
        )
    else:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            color=color,
            scale=scale,
            width=width,
            label=label,
            zorder=zorder,
            pivot="mid",
        )

    return ax


def project_vectors_and_positions_onto_cylinder(positions, vectors):
    n = positions.shape[0]
    projected_vectors = np.zeros((n, 2))
    projected_positions = np.zeros((n, 2))
    
    for i in range(n):
        x, y, z = positions[i]
        vector = vectors[i]

        # Compute the angular coordinate theta
        theta = np.arctan2(y, x)
        
        if theta < 0:
            theta += 2 * np.pi

        # Define the gauge directions
        gauge_theta = np.array([-y, x, 0])
        gauge_z = np.array([0, 0, 1])

        # Normalize the gauge direction in the theta direction
        gauge_theta /= np.linalg.norm(gauge_theta)

        # Define the gauge matrix
        gauge = np.column_stack((gauge_theta, gauge_z))

        # Project the vector onto the gauge directions
        projected_vector = np.dot(gauge.T, vector)

        projected_vectors[i] = projected_vector
        projected_positions[i] = [theta, z]

    return projected_positions, projected_vectors


def mesh_to_polyscope(mesh, wrap_x=True, wrap_y=True, reverse_x=False, reverse_y=False):
    n, m, _ = mesh.shape

    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    coords = jj + m * ii

    faces = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):

            c1 = [i, j]
            c2 = [(i + 1) % n, j]
            c3 = [(i + 1) % n, (j + 1) % m]
            c4 = [i, (j + 1) % m]

            # print(i, n)
            if (i == n - 1) and reverse_x:
                c2[1] = (-c2[1] - 2) % m
                c3[1] = (-c3[1] - 2) % m
                # c2[1] = (-c2[1] - int(m / 2) - 2) % m
                # c3[1] = (-c3[1] - int(m / 2) - 2) % m
            if (j == m - 1) and reverse_y:
                c3[0] = (-c3[0] - 2) % n
                c4[0] = (-c4[0] - 2) % n
                # c3[0] = (-c3[0] - int(n / 2) - 2) % n
                # c4[0] = (-c4[0] - int(n / 2) - 2) % n

            faces[i, j, 0] = coords[c1[0], c1[1]]
            faces[i, j, 1] = coords[c2[0], c2[1]]
            faces[i, j, 2] = coords[c3[0], c3[1]]
            faces[i, j, 3] = coords[c4[0], c4[1]]

    mesh_ = mesh.reshape(-1, 3)
    faces_ = faces.reshape(-1, 4)

    return mesh_, faces_


def cylinder_m_to_3d(M,radius=1):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = radius * np.sin(theta)
    c = radius * np.cos(theta)
    return np.stack([s, c, x], axis=-1)


def phase_to_coordinates(phase_angles, radius=1):
    x = radius * np.cos(phase_angles)
    y = radius * np.sin(phase_angles)
    return np.dstack((x, y))


def simulate_pendulum(n_points=15, n_traj=2, steps=2001, seed=0):
    
    np.random.seed(seed)

    positions = np.linspace(0, 2 * np.pi, n_points)
    momentums = np.linspace(-3, 3, n_points)
    positions, momentums = np.meshgrid(positions, momentums)
    positions = positions.flatten()[:, np.newaxis]
    momentums = momentums.flatten()[:, np.newaxis]
    grid = np.concatenate([positions, momentums], axis=-1)
    
    system = FrictionPendulumSystem(mass=0.1, length=2.0, friction=0.03)

    vector_field = system.dynamics_gradient_field(positions, momentums)
        
    initial_states = np.hstack([positions, momentums])
    
    rollouts = system.rollout(initial_states, steps)
    
    grid = np.array(grid, dtype=np.float64)
    vector_field = np.array(vector_field, dtype=np.float64)
    rollouts = np.array(rollouts, dtype=np.float64)
    
    return grid, vector_field, rollouts



def integrate_trajectory(y0, d, manifold_GP, vector_GP, h=5, len_t=400,):
    
    trajectory = [y0]
    vectors = []
    
    for i in range(len_t):
        
        # take the most recent manifold points
        y = trajectory[-1]  
        
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
    
    # combine trajectories for output
    trajectory = np.vstack(trajectory)[:-1,:]
    vectors = np.vstack(vectors)
    
    return trajectory, vectors


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