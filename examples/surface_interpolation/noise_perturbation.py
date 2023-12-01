#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
import RVGP
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances


n_eigenpairs=100
vertices, faces = load_mesh('bunny')
trials=10

np.random.seed(0)

results = []
for alpha in np.linspace(0.005,0.045,10):
    
    r2 = []
    for t in range(trials):
        print(t)
        
        sample_ind, _ = furthest_point_sampling(vertices, spacing=0.03)
        X = vertices[sample_ind]
        
        train_ind =  np.random.choice(np.arange(len(X)), size=int(0.8*len(X)))
        test_ind = set(range(len(X))) - set(train_ind)
        test_ind = list(test_ind)
        
        diam = pairwise_distances(vertices).max()
        X[test_ind] += alpha*diam*np.random.normal(size=X[test_ind].shape)
        
        d = RVGP.create_data_object(X, n_eigenpairs=n_eigenpairs)
        d.random_vector_field(seed=1)
        d.smooth_vector_field(t=100)
                
        vector_field_GP = RVGP.fit(d, train_ind=train_ind, noise_variance=0.001)
        f_pred_mean, _ = vector_field_GP.transform(d, test_ind)
        
        results.append([d.vectors[test_ind], f_pred_mean])
        
        pickle.dump(results, open('noise_perturbation.pkl','wb'))