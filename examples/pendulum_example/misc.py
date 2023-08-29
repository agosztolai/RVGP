#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from dynamical_systems import FrictionPendulumSystem


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
    
    initial_states = np.hstack([np.random.uniform(low=0,
                                                  high=2*np.pi, 
                                                  size=(n_traj,1)),
                                np.random.uniform(low=-3,
                                                  high=3, 
                                                  size=(n_traj,1))])
 
    rollouts = system.rollout(initial_states, steps)
    
    grid = np.array(grid, dtype=np.float64)
    vector_field = np.array(vector_field, dtype=np.float64)
    rollouts = np.array(rollouts, dtype=np.float64)
    
    return grid, vector_field, rollouts

grid, vector_field, rollouts= simulate_pendulum()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_vector_field(grid, vector_field, color="black", ax=ax, scale=300, width=0.004)
for i in range(rollouts.shape[0]):
    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.gca().set_aspect("equal")