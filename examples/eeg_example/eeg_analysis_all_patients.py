#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import os
from scipy.io import savemat

from eeg_utils import load_eeg_data, compute_error_time
from eeg_utils import interpolate_time_range
from eeg_utils import compute_vectorfield_features_time

np.random.seed(0)

#%%

def find_mat_files(directory):
    mat_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mat'):
                #mat_file_path = os.path.join(root, file)
                subdirectory_name = os.path.basename(root)
                #mat_files.append((mat_file_path, subdirectory_name))
                mat_files[subdirectory_name] = root +  '/'
    return mat_files

directory = '/media/robert/PortableSSD/ResearchProjects/RVGP/data/eeg_data/all_patients/' # Replace with your directory path
mat_files = find_mat_files(directory)



#%%

for patient_id, folder in mat_files.items():

#folder = mat_files['s1']

    # load data
    vectors_ds, vectors_si, vectors_gt, channel_locations_gt, channel_locations_ds = load_eeg_data(folder ,
                                                                                                   start_time=0,
                                                                                                   end_time=10000)
    
    # extract channel spatial locations
    X = channel_locations_gt[['X','Y','Z']].values.astype(float)
    
    # define channels for training and testing model
    train_idx = np.where(channel_locations_gt.index.isin(channel_locations_ds.index))[0]
    test_idx = np.arange(len(X), dtype=int).reshape(-1, 1)
    
    # extract vectors only
    f_real = vectors_gt[:,:,1:]  # taking x,y,z
    f_spline = vectors_si[:,:,1:] 
    f_down = vectors_ds[:,:,1:]    
   
    # run interpolation for a series of time points
    max_t = f_down.shape[0]
    gap = 1
    timepoints = list(range(0,max_t,gap))
    f_pred = interpolate_time_range(timepoints, X, f_down, train_idx, test_idx,)
    
    # compute errors
    l2_error_ds = compute_error_time(f_pred, f_real, test_idx)
    l2_error_si = compute_error_time(f_spline, f_real, test_idx)
    
    # compute vector field stats
    div = {}
    curl = {}
    div['gt'], curl['gt'] = compute_vectorfield_features_time(timepoints, X, f_real)
    div['ds'], curl['ds'] = compute_vectorfield_features_time(timepoints, X, f_pred)
    div['si'], curl['si'] = compute_vectorfield_features_time(timepoints, X, f_spline)
    
    # compile and save
    results = {}
    results['f_pred'] = {'f_pred':f_pred}
    results['divergence'] = div
    results['curl'] = curl
    results['error'] = {'rvgp': l2_error_ds, 'spline': l2_error_si}
    
    savemat(folder + 'results.mat', mdict=results, oned_as='row')



