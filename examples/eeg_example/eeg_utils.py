
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

from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree, KDTree
import random 

def interpolate_time_range(timepoints, X, f, train_idx, test_idx, predict_manifold=False):
    
    if predict_manifold: # use only the channels with vectors to approximate manifold
        d = data(X[train_idx,:], n_eigenpairs=30)
    else: # us all channels to approximate latent manifold
        d = data(X, n_eigenpairs=10)
    
    f_pred = np.zeros([len(timepoints), X.shape[0], X.shape[1]])
    for t, time in enumerate(tqdm(timepoints)):
        d.vectors = f[int(time),:,:]        
        f_pred[t,:,:] = interpolate_timepoint(d,
                                              train_idx,
                                              test_idx,
                                              project=False,
                                              predict_manifold=predict_manifold)
    
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
                          predict_manifold=False,
                          project=True,
                          t=0,
                          plot=False,
                          dim_emb=3,
                          dim_man=2,
                          n_eigenpairs = 50
                          ):
    
    # fit GP to manifold and then find manifold points closest to ground truth coordinates
    if predict_manifold:
        
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
        
        # generate new positional encodings based on local proximity 
        sampled = generate_new_points(positional_encoding, num_new_points=2000, k=5)
        
        # predict xyz points - this isn't perfect and ends up with some inside the head
        pred_xyz, _ = manifold_GP.predict_f(sampled)
        



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

def sample_evenly_from_manifold(points, num_samples=32):
    """
    Sample points evenly from a 3D manifold using K-means clustering,
    and return both the sampled points and the indices of these points in the original array.

    :param points: numpy array of points on the manifold
    :param num_samples: number of points to sample, default is 32
    :return: tuple of numpy array of sampled points and numpy array of their indices
    """
    # Ensure that the number of clusters does not exceed the number of points
    num_clusters = min(num_samples, len(points))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(points)

    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Find the indices of the closest points in 'points' to each centroid
    _, indices = tree.query(centroids, k=1)

    # Initialize arrays for additional samples and their indices
    additional_samples = []
    additional_indices = []

    # If the desired number of samples is greater than the number of clusters,
    # sample additional points randomly from each cluster
    if num_samples > num_clusters:
        labels = kmeans.labels_
        for i in range(num_clusters):
            cluster_points_indices = np.where(labels == i)[0]
            n_samples = int(np.ceil(num_samples / num_clusters)) - 1  # Subtract 1 for the centroid
            sampled_indices = np.random.choice(cluster_points_indices, n_samples, replace=True)
            additional_indices.extend(sampled_indices)
            additional_samples.extend(points[sampled_indices])

    # Combine centroid points and additional samples
    sampled_points = np.concatenate([centroids, points[additional_indices][:num_samples - num_clusters]])
    sampled_indices = np.concatenate([indices, additional_indices])[:num_samples]
    
    # Sort the indices and reorder the points accordingly
    sorted_indices = np.argsort(sampled_indices)
    sampled_indices = sampled_indices[sorted_indices]
    sampled_points = sampled_points[sorted_indices]

    # Convert indices to integers
    sampled_indices = sampled_indices.astype(int)
    return sampled_points, sampled_indices

def generate_new_points(data, num_new_points=10, k=5):
    # Initialize KDTree
    tree = KDTree(data)

    new_points = []
    for _ in range(num_new_points):
        # Randomly select a reference point
        ref_point_index = np.random.randint(data.shape[0])
        ref_point = data[ref_point_index]

        # Find k-nearest neighbors (including the point itself)
        distances, indices = tree.query(ref_point, k=k+1)

        # Exclude the point itself from the neighbors
        indices = indices[indices != ref_point_index]

        # Ensure at least 1 neighbor is always included
        num_neighbors_to_average = random.randint(1, min(len(indices), 2))
        selected_indices = np.random.choice(indices, num_neighbors_to_average, replace=False)

        # Compute the average point
        average_point = np.mean(data[selected_indices], axis=0)

        new_points.append(average_point)

    return np.array(new_points)


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