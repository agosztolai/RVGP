#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gpflow

from scipy.io import loadmat
import pandas as pd

from ptu_dijkstra import connections, tangent_frames

from RVGP.geometry import (furthest_point_sampling, 
                           manifold_graph, 
                           compute_laplacian,
                           compute_connection_laplacian,
                           compute_spectrum,
                           sample_from_convex_hull,
                           project_to_manifold, 
                           project_to_local_frame,
                           node_eigencoords
                           )
from RVGP.smoothing import vector_diffusion
from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel

np.random.seed(0)

#%%
# =============================================================================
# Define manifold points
# =============================================================================

folder = '/media/robert/PortableSSD/ResearchProjects/RVGP/data/eeg_data/'

eeg_data = loadmat(folder + 'obj_flows_original.mat')['obj_flows'][0][0]

# original channel locations
channel_locations = pd.DataFrame(eeg_data[4][0])
channel_locations = channel_locations[['labels','X','Y','Z']]
channel_locations = channel_locations.explode(list(channel_locations.columns))
channel_locations = channel_locations.explode(['X','Y','Z'])
X = channel_locations[['X','Y','Z']].values.astype(float)


# extract downsampled data
vn = eeg_data[0][0,:]
vx = eeg_data[1][0,:]
vy = eeg_data[2][0,:]
vz = eeg_data[3][0,:]
vectors = np.vstack([vn,vx,vy,vz]).T

dim_emb = 3


# =============================================================================
# Fit graph, tangent frames and connections
# =============================================================================
G = manifold_graph(X)
dim_man = 2
gauges, Sigma = tangent_frames(X, G, dim_man, 10)
R = connections(gauges, G, dim_man)

# =============================================================================
# Eigendecomposition of connection Laplacian
# =============================================================================
L = compute_laplacian(G)
Lc = compute_connection_laplacian(G, R)

n_eigenpairs = 50
evals_Lc, evecs_Lc = compute_spectrum(Lc, n_eigenpairs) # U\Lambda U^T

#rather than U, take TU, where T is the local gauge
evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man,n_eigenpairs)
evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)

# =============================================================================
# Define vector field over manifold and smooth it using vector diffusion
# =============================================================================
# f = np.random.uniform(size=(len(X), 3))-.5
# f = project_to_manifold(f, gauges[...,:2])
# f /= np.linalg.norm(f, axis=1, keepdims=True)

f = vectors[:,1:]

ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='b', length=0.3)

t=10
f = project_to_local_frame(f, gauges)
#f = vector_diffusion(f, t, L=L, Lc=Lc, method="matrix_exp")
f = project_to_local_frame(f, gauges, reverse=True)

ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)
ax.axis('equal')

# =============================================================================
# Custom kernel
# =============================================================================

kernel = ManifoldKernel((evecs_Lc, evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)

#%%

# =============================================================================
# Train GP for manifold
# =============================================================================
x_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs) # evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)
y_train = X

manifold_GP = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP.training_loss, manifold_GP.trainable_variables)

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
node_train = np.arange(0,256,8)
node_test = np.arange(len(X), dtype=int).reshape(-1, 1)
x_train_f = node_eigencoords(node_train, evecs_Lc, dim_emb)
f_train = f[node_train.flatten()]
x_test = node_eigencoords(node_test, evecs_Lc, dim_emb)
vector_field_GP = gpflow.models.GPR((x_train_f, f_train.reshape(-1, 1)), kernel=kernel, noise_variance=0.001)

# optimize_GPR(vector_field_GP, 10000) #this is an alternative using gradient descent
# gpflow.utilities.print_summary(vector_field_GP)    
opt = gpflow.optimizers.Scipy()
opt.minimize(vector_field_GP.training_loss, vector_field_GP.trainable_variables)

# =============================================================================
# Test performance
# =============================================================================
f_pred_mean, _ = vector_field_GP.predict_f(x_test) 
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

l2_error = np.linalg.norm(f.ravel() - f_pred_mean.ravel()) / len(f.ravel())
print("Relative l2 error is {}".format(l2_error))

ax = graph(G)
ax.quiver(X[node_train,0], X[node_train,1], X[node_train,2], f_train[:,0], f_train[:,1], f_train[:,2], color='g', length=0.3)
#ax.quiver(X[:,0], X[:,1], X[:,2], f_pred_mean[:,0], f_pred_mean[:,1], f_pred_mean[:,2], color='r', length=0.3)
ax.axis('equal')


ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)
ax.quiver(X[:,0], X[:,1], X[:,2], f_pred_mean[:,0], f_pred_mean[:,1], f_pred_mean[:,2], color='r', length=0.3)
ax.axis('equal')


#%%

import plotly.graph_objects as go

scale = 2

fig = go.Figure(data = go.Cone(
    x=X[:,0],
    y=X[:,1],
    z=X[:,2],
    u=f[:,0]*scale,
    v=f[:,1]*scale,
    w=f[:,2]*scale,
    colorscale='Blues',
    sizemode="absolute",
    sizeref=40))

fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

fig.write_html("real_eeg_flow.html")


scale = 2

fig = go.Figure(data = go.Cone(
    x=X[:,0],
    y=X[:,1],
    z=X[:,2],
    u=f_pred_mean[:,0]*scale,
    v=f_pred_mean[:,1]*scale,
    w=f_pred_mean[:,2]*scale,
    colorscale='Blues',
    sizemode="absolute",
    sizeref=40))

fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

fig.write_html("predicted_eeg_flow.html")
