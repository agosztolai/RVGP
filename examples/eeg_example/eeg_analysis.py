#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import mne

import scipy.stats as st
from scipy.io import savemat

from eeg_utils import load_eeg_data, interpolate_timepoint, compute_error
from eeg_utils import plot_interpolation_topomap, interpolate_time_range
from eeg_utils import compute_vectorfield_features, compute_vectorfield_features_time

np.random.seed(0)

#%%

folder = '/media/robert/PortableSSD/ResearchProjects/RVGP/data/eeg_data/'

# load data
vectors_ds, vectors_si, vectors_gt, channel_locations_gt, channel_locations_ds = load_eeg_data(folder,
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

# run interpolation for a single time point
t = 3000
f_pred = interpolate_timepoint(X, f_down[t,:,:], train_idx, test_idx, project=False)
l2_error = compute_error(f_pred, f_real[t,:,:], test_idx, verbose=True)
l2_error = compute_error(f_spline[t,:,:], f_real[t,:,:], test_idx, verbose=True)

# compute div and curl of vector field
div_gt, curl_gt = compute_vectorfield_features(X, f_real[t,:,:])
div_ds, curl_ds = compute_vectorfield_features(X, f_pred)
div_si, curl_si = compute_vectorfield_features(X, f_spline[t,:,:])

# plot interpolation topomap
plot_interpolation_topomap(X, div_gt.reshape(-1,1), channel_locations_gt)
plot_interpolation_topomap(X, div_ds.reshape(-1,1), channel_locations_gt)
plot_interpolation_topomap(X, div_si.reshape(-1,1), channel_locations_gt)

plot_interpolation_topomap(X, np.linalg.norm(curl_gt, axis=1).reshape(-1,1), channel_locations_gt)
plot_interpolation_topomap(X, np.linalg.norm(curl_ds, axis=1).reshape(-1,1), channel_locations_gt)
plot_interpolation_topomap(X, np.linalg.norm(curl_si, axis=1).reshape(-1,1), channel_locations_gt)


# run interpolation for a series of time points
max_t = f_down.shape[0]
gap = 1
timepoints = list(range(0,max_t,gap))
f_pred = interpolate_time_range(timepoints, X, f_down, train_idx, test_idx,)


# compute vector field stats
div = {}
curl = {}
div['gt'], curl['gt'] = compute_vectorfield_features_time(timepoints, X, f_real)
div['ds'], curl['ds'] = compute_vectorfield_features_time(timepoints, X, f_pred)
div['si'], curl['si'] = compute_vectorfield_features_time(timepoints, X, f_spline)

savemat('f_pred.mat', mdict={'f_pred':f_pred}, oned_as='row')
savemat('divergence.mat', mdict=div, oned_as='row')
savemat('curl.mat', mdict=curl, oned_as='row')


