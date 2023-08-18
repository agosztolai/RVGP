#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gpflow

from scipy.io import loadmat
import scipy.sparse as sp
import pandas as pd

import matplotlib.pyplot as plt

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

eeg_data = loadmat(folder + 'obj_flows.mat')['obj_flows'][0][0]


#%%

timepoint = 5000

# extract downsampled data
channel_locations_ds = pd.DataFrame(eeg_data[0][0][0][0][0][0][1][0])
channel_locations_ds = channel_locations_ds[['labels','X','Y','Z']]
channel_locations_ds = channel_locations_ds.explode(list(channel_locations_ds.columns))
channel_locations_ds = channel_locations_ds.explode(['X','Y','Z'])
channel_locations_ds = channel_locations_ds.set_index('labels')
vn = eeg_data[0][0][0][0][0][0][0][0][0][0][timepoint,:]
vx = eeg_data[0][0][0][0][0][0][0][0][0][1][timepoint,:]
vy = eeg_data[0][0][0][0][0][0][0][0][0][2][timepoint,:]
vz = eeg_data[0][0][0][0][0][0][0][0][0][3][timepoint,:]
vectors_ds = np.vstack([vn,vx,vy,vz]).T

# extract spline interpolated data
channel_locations_si = pd.DataFrame(eeg_data[0][0][0][1][0][0][1][0])
channel_locations_si = channel_locations_si[['labels','X','Y','Z']]
channel_locations_si = channel_locations_si.explode(list(channel_locations_si.columns))
channel_locations_si = channel_locations_si.explode(['X','Y','Z'])
channel_locations_si = channel_locations_si.set_index('labels')
vn = eeg_data[0][0][0][1][0][0][0][0][0][0][timepoint,:]
vx = eeg_data[0][0][0][1][0][0][0][0][0][1][timepoint,:]
vy = eeg_data[0][0][0][1][0][0][0][0][0][2][timepoint,:]
vz = eeg_data[0][0][0][1][0][0][0][0][0][3][timepoint,:]
vectors_si = np.vstack([vn,vx,vy,vz]).T

# extract ground truth data
channel_locations_gt = pd.DataFrame(eeg_data[0][0][0][2][0][0][1][0])
channel_locations_gt = channel_locations_gt[['labels','X','Y','Z']]
channel_locations_gt = channel_locations_gt.explode(list(channel_locations_gt.columns))
channel_locations_gt = channel_locations_gt.explode(['X','Y','Z'])
channel_locations_gt = channel_locations_gt.set_index('labels')
vn = eeg_data[0][0][0][2][0][0][0][0][0][0][timepoint,:]
vx = eeg_data[0][0][0][2][0][0][0][0][0][1][timepoint,:]
vy = eeg_data[0][0][0][2][0][0][0][0][0][2][timepoint,:]
vz = eeg_data[0][0][0][2][0][0][0][0][0][3][timepoint,:]
vectors_gt = np.vstack([vn,vx,vy,vz]).T

#%%

dim_emb = 3
dim_man = 2

#train_channels = [*channel_locations_ds.index]
#test_channels = [*channel_locations_gt.index]

X_down = channel_locations_ds[['X','Y','Z']].values.astype(float)
X_full = channel_locations_gt[['X','Y','Z']].values.astype(float)
train_idx = np.where(channel_locations_gt.index.isin(channel_locations_ds.index))[0]
test_idx = np.setdiff1d(np.arange(0,X_full.shape[0]),train_idx)


#%%


# =============================================================================
# Fit graph, tangent frames and connections
# =============================================================================
G = manifold_graph(X_full)
gauges, Sigma = tangent_frames(X_full, G, dim_man, 10)
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
#f_down = np.random.uniform(size=(len(X_full), 3))-.5
# f = project_to_manifold(f, gauges[...,:2])
#f_down /= np.linalg.norm(f_down, axis=1, keepdims=True)

f_down = vectors_ds[:,1:]

ax = graph(G)
ax.quiver(X_down[:,0], X_down[:,1], X_down[:,2], f_down[:,0], f_down[:,1], f_down[:,2], color='b', length=0.3)

#t=20
#Lc_idx = np.sort(np.hstack([train_idx*2, train_idx*2+1]))
#Lc_ds = sp.bsr_matrix(Lc.A[Lc_idx,:][:,Lc_idx])
#L_ds = L[train_idx,:][:, train_idx]

#f_down = project_to_local_frame(f_down, gauges[train_idx,:,:])
#f_down = vector_diffusion(f_down, t, L=L_ds , Lc=Lc_ds, method="matrix_exp")
#f_down = project_to_local_frame(f_down, gauges[train_idx,:,:], reverse=True)

#ax.quiver(X_down[:,0], X_down[:,1], X_down[:,2], f_down[:,0], f_down[:,1], f_down[:,2], color='g', length=0.3)
#ax.axis('equal')

# =============================================================================
# Custom kernel
# =============================================================================

kernel = ManifoldKernel((evecs_Lc, evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)




#%%

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================

#node_test = np.arange(len(X_full), dtype=int).reshape(-1, 1)

x_train_f = node_eigencoords(train_idx, evecs_Lc, dim_emb)
vector_field_GP = gpflow.models.GPR((x_train_f, f_down.reshape(-1, 1)), kernel=kernel, noise_variance=0.001)


opt = gpflow.optimizers.Scipy()
opt.minimize(vector_field_GP.training_loss, vector_field_GP.trainable_variables)



#%%

# =============================================================================
# Test performance
# =============================================================================

f_test = vectors_gt[:,1:]

test_idx = np.arange(len(X_full), dtype=int).reshape(-1, 1)
x_test_f = node_eigencoords(test_idx, evecs_Lc, dim_emb)

f_pred_mean, _ = vector_field_GP.predict_f(x_test_f) 
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

l2_error = np.linalg.norm(f_test[test_idx,:].ravel() - f_pred_mean[test_idx,:].ravel()) / len(f_test[test_idx,:].ravel())
print("Relative l2 error is {}".format(l2_error))


ax = graph(G)
ax.quiver(X_full[:,0], X_full[:,1], X_full[:,2], f_test[:,0], f_test[:,1], f_test[:,2], color='g', length=0.3)
ax.quiver(X_full[:,0], X_full[:,1], X_full[:,2], f_pred_mean[:,0], f_pred_mean[:,1], f_pred_mean[:,2], color='r', length=0.3)
ax.axis('equal')


# =============================================================================
# Spline interpolation performance
# =============================================================================

f_spline = vectors_si[:,1:]


l2_error = np.linalg.norm(f_test[test_idx,:].ravel() - f_spline[test_idx,:].ravel()) / len(f_test[test_idx,:].ravel())
print("Relative l2 error is {}".format(l2_error))


ax = graph(G)
ax.quiver(X_full[:,0], X_full[:,1], X_full[:,2], f_test[:,0], f_test[:,1], f_test[:,2], color='g', length=0.3)
ax.quiver(X_full[:,0], X_full[:,1], X_full[:,2], f_spline[:,0], f_spline[:,1], f_spline[:,2], color='r', length=0.3)
ax.axis('equal')

#%%


import numpy as np
from scipy.spatial import KDTree

import mne
from mne.viz import plot_topomap


def compute_divergence(positions, vectors, k=5):

    # Build a KD-tree for efficient nearest neighbour search
    tree = KDTree(positions)       
    
    div = np.zeros([positions.shape[0]])
    curl = np.zeros([positions.shape[0],3])
    
    for j in range(positions.shape[0]):
        point = positions[j,:]
        distances, indices = tree.query(point, k+1)  # +1 because the point itself is included
        
        # Compute the divergence estimate
        divergence_estimate = 0
        curl_estimate = np.zeros(3)
        for i in range(1, len(indices)):  # start from 1 to skip the point itself
            delta_position = positions[indices[i]] - point
            delta_vector = vectors[indices[i]] - vectors[0]
            divergence_estimate += np.dot(delta_vector, delta_position) / np.linalg.norm(delta_position)**2
            curl_estimate += np.cross(delta_vector, delta_position) / np.linalg.norm(delta_position)**2
            
        divergence_estimate /= k  # normalize by k to get the average
        curl_estimate /= k  # normalize by k to get the average
                
        div[j] = divergence_estimate
        curl[j,:] = curl_estimate
   
    return div, curl
    
div_gt, curl_gt = compute_divergence(X_full, f_test)
div_ds, curl_ds = compute_divergence(X_full, f_pred_mean)
div_si, curl_si = compute_divergence(X_full, f_spline)


montage = mne.channels.make_dig_montage(ch_pos=dict(zip(list(channel_locations_gt.index), X_full)), coord_frame='head')
info = mne.create_info(list(channel_locations_gt.index),sfreq=256, ch_types='eeg')
info.set_montage(montage)

# plot div
plt.figure()
evoked = mne.EvokedArray(div_gt.reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(div_ds.reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(div_si.reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)


# plot curl in x
plt.figure()
evoked = mne.EvokedArray(curl_gt[:,0].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(curl_ds[:,0].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(curl_si[:,0].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)


# plot curl in y
plt.figure()
evoked = mne.EvokedArray(curl_gt[:,1].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(curl_ds[:,1].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(curl_si[:,1].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)


# plot curl in z
plt.figure()
evoked = mne.EvokedArray(curl_gt[:,2].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(curl_ds[:,2].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

plt.figure()
evoked = mne.EvokedArray(curl_si[:,2].reshape(-1,1), info)
evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)


#%%

import plotly.graph_objects as go

scale = 1

fig = go.Figure(data = go.Cone(
    x=X_full[:,0],
    y=X_full[:,1],
    z=X_full[:,2],
    u=f_test[:,0]*scale,
    v=f_test[:,1]*scale,
    w=f_test[:,2]*scale,
    colorscale='Blues',
    sizemode="absolute",
    sizeref=40))

fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

fig.write_html("real_eeg_flow.html")



scale = 2

fig = go.Figure(data = go.Cone(
    x=X_full[:,0],
    y=X_full[:,1],
    z=X_full[:,2],
    u=f_pred_mean[:,0]*scale,
    v=f_pred_mean[:,1]*scale,
    w=f_pred_mean[:,2]*scale,
    colorscale='Blues',
    sizemode="absolute",
    sizeref=40))

fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

fig.write_html("predicted_eeg_flow.html")


scale = 1

fig = go.Figure(data = go.Cone(
    x=X_full[:,0],
    y=X_full[:,1],
    z=X_full[:,2],
    u=f_spline[:,0]*scale,
    v=f_spline[:,1]*scale,
    w=f_spline[:,2]*scale,
    colorscale='Blues',
    sizemode="absolute",
    sizeref=40))

fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

fig.write_html("spline_eeg_flow.html")
