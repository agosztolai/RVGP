#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
import RVGP
import numpy as np
import pickle

n_eigenpairs=300
vertices, faces = load_mesh('bunny')
trials=10

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, spacing=0.01)
X = vertices[sample_ind]
d = RVGP.create_data_object(X, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)

np.random.seed(0)

evals_L, evecs_Lc = d.evals_L, d.evecs_L
evals_Lc, evecs_Lc = d.evals_Lc, d.evecs_Lc

results, results_lap = [], []
for k in [1,2,5,10,100,200,300]:

    for t in range(trials):
        print(t)
        train_ind =  np.random.choice(np.arange(len(X)), size=int(0.8*len(X)))
        test_ind = set(range(len(X))) - set(train_ind)
        test_ind = list(test_ind)
        test_f = d.vectors[test_ind]
        
        d.evals_L, d.evecs_L = evals_L[:k], evecs_Lc[:,:k]
        d.evals_Lc, d.evecs_Lc = evals_Lc[:k], evecs_Lc[:,:k]
        
        #with manifold kernel (connection Laplacian)
        vector_field_GP = RVGP.fit(d, train_ind=train_ind, noise_variance=0.001)
        f_pred_mean, _ = vector_field_GP.transform(d, test_ind)
        
        #with RBF kernel, treating entries channelwise
        vector_field_GP_lap = RVGP.fit(d, kernel='rbf', noise_variance=0.001)
        f_pred_mean_lap, _ = vector_field_GP.transform(d, test_ind)
        
        results.append([test_f, f_pred_mean])
        results_lap.append([test_f, f_pred_mean_lap])
        
        pickle.dump(results, open('ablation_eigenvector_results.pkl','wb'))
        pickle.dump(results_lap, open('ablation_eigenvector_results_lap.pkl','wb'))