
from scipy.io import loadmat
import scipy.sparse as sp
import pandas as pd
import numpy as np


import mne

from RVGP.geometry import (compute_laplacian,
                           project_to_local_frame,
                           )
from RVGP.smoothing import vector_diffusion
from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel

from RVGP import data, train_gp

from scipy.spatial import KDTree
from tqdm import tqdm


def interpolate_time_range(timepoints, X, f, train_idx, test_idx,):
    
    d = data(X, n_eigenpairs=10)
    
    f_pred = np.zeros([len(timepoints), X.shape[0], X.shape[1]])
    for t, time in enumerate(tqdm(timepoints)):
        d.vectors = f[int(time),:,:]
        f_pred[t,:,:] = interpolate_timepoint(d, train_idx, test_idx, project=False)
    
    return f_pred

def compute_vectorfield_features_time(timepoints, positions, vectors):
    
    div = np.zeros([len(timepoints), positions.shape[0]])
    curl = np.zeros([len(timepoints), positions.shape[0],3])
    for t in timepoints:
        div[t,:], curl[t,:,:] = compute_vectorfield_features(positions, vectors[t,:,:], k=5)    
    
    return div, curl

def compute_vectorfield_features(positions, vectors, k=5):
    
    # Normalize the vectors (interested in geometry of vector field - not magnitude)
    magnitudes = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / magnitudes[:, np.newaxis]

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
            delta_vector = normalized_vectors[indices[i]] - normalized_vectors[0]
            divergence_estimate += np.dot(delta_vector, delta_position) / np.linalg.norm(delta_position)**2
            curl_estimate += np.cross(delta_vector, delta_position) / np.linalg.norm(delta_position)**2
            
        divergence_estimate /= k  # normalize by k to get the average
        curl_estimate /= k  # normalize by k to get the average
                
        div[j] = divergence_estimate
        curl[j,:] = curl_estimate
        
    # take magnitude of curl
    #curl_magnitude = np.linalg.norm(curl, axis=1)   
    
    return div, curl


def interpolate_timepoint(d,
                          train_idx,
                          test_idx,
                          project=True,
                          t=0,
                          plot=False,
                          dim_emb=3,
                          dim_man=2,
                          n_eigenpairs = 50
                          ):

       
    if project:
        d.vectors = project_to_local_frame(d.vectors, d.gauges[train_idx,:,:])        
        
        if t>0:
            Lc_idx = np.sort(np.hstack([train_idx*2, train_idx*2+1]))
            Lc_ = sp.bsr_matrix(d.Lc.A[Lc_idx,:][:,Lc_idx])            
            L = compute_laplacian(d.G)
            L = L[train_idx,:][:, train_idx]            
            d.vectors = vector_diffusion(d.vectors, t, L=L , Lc=Lc_, method="matrix_exp")
            
        d.vectors = project_to_local_frame(d.vectors, d.gauges[train_idx,:,:], reverse=True)
    
    # Custom kernel   
    kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                            nu=3/2, 
                            kappa=5, 
                            sigma_f=1)  
    
    # Train GP for vector field over manifold
    x_train = d.evecs_Lc.reshape(d.n, -1)[train_idx,:]    
    sp_to_vector_field_gp = train_gp(x_train, 
                                     d.vectors,
                                     dim=d.vertices.shape[1],
                                     epochs=100,
                                     kernel=kernel,
                                     noise_variance=0.001,
                                     compute_error=False)

           
    # Test performance   
    x_test = d.evecs_Lc.reshape(d.n, -1)[test_idx,:]      
    f_pred, _ = sp_to_vector_field_gp.predict_f(x_test.reshape(len(test_idx)*d.vertices.shape[1], -1))
    f_pred = f_pred.numpy().reshape(len(test_idx), -1)    
    
    return f_pred


def compute_error(f_pred, f_real, test_idx, plot=False, verbose=False):    
    l2_error = np.linalg.norm(f_real[test_idx,:].ravel() - f_pred[test_idx,:].ravel()) / len(f_real[test_idx,:].ravel())
    if verbose:
        print("Relative l2 error is {}".format(l2_error)) 
    return l2_error

def compute_error_time(f_pred, f_real, test_idx):      
    squared_diffs = (f_real - f_pred) ** 2
    sum_squared_diffs = np.einsum('ijk->i', squared_diffs)
    l2_errors = np.sqrt(sum_squared_diffs)
    return l2_errors


def plot_predictions(X, f_pred, f_real, G):
    ax = graph(G)
    ax.quiver(X[:,0], X[:,1], X[:,2], f_real[:,0], f_real[:,1], f_real[:,2], color='g', length=0.3)
    ax.quiver(X[:,0], X[:,1], X[:,2], f_pred[:,0], f_pred[:,1], f_pred[:,2], color='r', length=0.3)
    ax.axis('equal')          
    return

def plot_interpolation_topomap(X, data, channel_locations):

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(list(channel_locations.index), X)), coord_frame='head')
    info = mne.create_info(list(channel_locations.index), sfreq=250, ch_types='eeg')
    info.set_montage(montage)

    # plot div
    evoked = mne.EvokedArray(data, info)
    evoked.plot_topomap(ch_type='eeg',times=[0], size=4, res=256)

def load_eeg_data(folder, start_time=0, end_time=-1):
    
    eeg_data = loadmat(folder + 'obj_flows.mat')['obj_flows'][0][0]
    
    # extract downsampled data
    channel_locations_ds = pd.DataFrame(eeg_data[0][0][0][0][0][0][1][0])
    channel_locations_ds = channel_locations_ds[['labels','X','Y','Z']]
    channel_locations_ds = channel_locations_ds.explode(list(channel_locations_ds.columns))
    channel_locations_ds = channel_locations_ds.explode(['X','Y','Z'])
    channel_locations_ds = channel_locations_ds.set_index('labels')
    vn = eeg_data[0][0][0][0][0][0][0][0][0][0][start_time:end_time,:]
    vx = eeg_data[0][0][0][0][0][0][0][0][0][1][start_time:end_time,:]
    vy = eeg_data[0][0][0][0][0][0][0][0][0][2][start_time:end_time,:]
    vz = eeg_data[0][0][0][0][0][0][0][0][0][3][start_time:end_time,:]
    vectors_ds = np.dstack([vn,vx,vy,vz])
    
    # extract spline interpolated data
    channel_locations_si = pd.DataFrame(eeg_data[0][0][0][1][0][0][1][0])
    channel_locations_si = channel_locations_si[['labels','X','Y','Z']]
    channel_locations_si = channel_locations_si.explode(list(channel_locations_si.columns))
    channel_locations_si = channel_locations_si.explode(['X','Y','Z'])
    channel_locations_si = channel_locations_si.set_index('labels')
    vn = eeg_data[0][0][0][1][0][0][0][0][0][0][start_time:end_time,:]
    vx = eeg_data[0][0][0][1][0][0][0][0][0][1][start_time:end_time,:]
    vy = eeg_data[0][0][0][1][0][0][0][0][0][2][start_time:end_time,:]
    vz = eeg_data[0][0][0][1][0][0][0][0][0][3][start_time:end_time,:]
    vectors_si = np.dstack([vn,vx,vy,vz])
    
    # extract ground truth data
    channel_locations_gt = pd.DataFrame(eeg_data[0][0][0][2][0][0][1][0])
    channel_locations_gt = channel_locations_gt[['labels','X','Y','Z']]
    channel_locations_gt = channel_locations_gt.explode(list(channel_locations_gt.columns))
    channel_locations_gt = channel_locations_gt.explode(['X','Y','Z'])
    channel_locations_gt = channel_locations_gt.set_index('labels')
    vn = eeg_data[0][0][0][2][0][0][0][0][0][0][start_time:end_time,:]
    vx = eeg_data[0][0][0][2][0][0][0][0][0][1][start_time:end_time,:]
    vy = eeg_data[0][0][0][2][0][0][0][0][0][2][start_time:end_time,:]
    vz = eeg_data[0][0][0][2][0][0][0][0][0][3][start_time:end_time,:]
    vectors_gt = np.dstack([vn,vx,vy,vz])
    
    
    return vectors_ds, vectors_si, vectors_gt, channel_locations_gt, channel_locations_ds