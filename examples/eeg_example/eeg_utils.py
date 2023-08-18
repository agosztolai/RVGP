
from scipy.io import loadmat
import scipy.sparse as sp
import pandas as pd
import numpy as np


import gpflow
import mne


from ptu_dijkstra import connections, tangent_frames

from RVGP.geometry import (manifold_graph, 
                           compute_laplacian,
                           compute_connection_laplacian,
                           compute_spectrum,
                           project_to_manifold, 
                           project_to_local_frame,
                           node_eigencoords
                           )
from RVGP.smoothing import vector_diffusion
from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel

from scipy.spatial import KDTree
from tqdm import tqdm


def interpolate_time_range(timepoints, X, f, train_idx, test_idx,):
    
    f_pred = np.zeros([len(timepoints), X.shape[0], X.shape[1]])
    for t, time in enumerate(tqdm(timepoints)):
        f_pred[t,:,:] = interpolate_timepoint(X, f[int(time),:,:], train_idx, test_idx, project=False)
    
    return f_pred

def compute_vectorfield_features_time(timepoints, positions, vectors):
    
    div = np.zeros([len(timepoints), positions.shape[0]])
    curl = np.zeros([len(timepoints), positions.shape[0],3])
    for t in timepoints:
        div[t,:], curl[t,:,:] = compute_vectorfield_features(positions, vectors[t,:,:], k=5)    
    
    return div, curl

def compute_vectorfield_features(positions, vectors, k=5):

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
        
    # take magnitude of curl
    #curl_magnitude = np.linalg.norm(curl, axis=1)   
    
    return div, curl


def interpolate_timepoint(X,
                          f,
                          train_idx,
                          test_idx,
                          project=True,
                          t=0,
                          plot=False,
                          dim_emb=3,
                          dim_man=2,
                          n_eigenpairs = 50
                          ):

    # Fit graph, tangent frames and connections
    G = manifold_graph(X)
    gauges, Sigma = tangent_frames(X, G, dim_man, 10)
    R = connections(gauges, G, dim_man)

    # Eigendecomposition of connection Laplacian
    Lc = compute_connection_laplacian(G, R)

    n_eigenpairs = 50
    evals_Lc, evecs_Lc = compute_spectrum(Lc, n_eigenpairs) # U\Lambda U^T
    
    #rather than U, take TU, where T is the local gauge
    evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man,n_eigenpairs)
    evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
    evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)
       
    if project:
        f = project_to_local_frame(f, gauges[train_idx,:,:])        
        
        if t>0:
            Lc_idx = np.sort(np.hstack([train_idx*2, train_idx*2+1]))
            Lc_ = sp.bsr_matrix(Lc.A[Lc_idx,:][:,Lc_idx])            
            L = compute_laplacian(G)
            L = L[train_idx,:][:, train_idx]            
            f = vector_diffusion(f, t, L=L , Lc=Lc_, method="matrix_exp")
            
        f = project_to_local_frame(f, gauges[train_idx,:,:], reverse=True)
    
    # Custom kernel    
    kernel = ManifoldKernel((evecs_Lc, evals_Lc), 
                            nu=3/2, 
                            kappa=5, 
                            sigma_f=1)


    # Train GP for vector field over manifold
    x_train_f = node_eigencoords(train_idx, evecs_Lc, dim_emb)
    vector_field_GP = gpflow.models.GPR((x_train_f, f.reshape(-1, 1)), kernel=kernel, noise_variance=0.001)
        
    opt = gpflow.optimizers.Scipy()
    opt.minimize(vector_field_GP.training_loss, vector_field_GP.trainable_variables)
           
    # Test performance   
    test_idx = np.arange(len(X), dtype=int).reshape(-1, 1)
    x_test_f = node_eigencoords(test_idx, evecs_Lc, dim_emb)    
    f_pred, _ = vector_field_GP.predict_f(x_test_f) 
    f_pred = f_pred.numpy().reshape(-1,dim_emb)    
    
    return f_pred


def compute_error(f_pred, f_real, test_idx, plot=False, verbose=False):    
    l2_error = np.linalg.norm(f_real[test_idx,:].ravel() - f_pred[test_idx,:].ravel()) / len(f_real[test_idx,:].ravel())
    if verbose:
        print("Relative l2 error is {}".format(l2_error)) 
    return l2_error

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