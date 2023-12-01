#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import RVGP
from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
import polyscope as ps
import numpy as np
from sklearn.metrics import pairwise_distances

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=100
n_neighbors=10
vertices, faces = load_mesh('bunny') #see /examples/data for more objects

# =============================================================================
# Subsample 
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, spacing=0.05)
X = vertices[sample_ind]

train_ind =  np.random.choice(np.arange(len(X)), size=int(0.5*len(X)))
test_ind = [i for i in range(len(X)) if i not in train_ind]
# =============================================================================
# Add noise
# =============================================================================
# diam = pairwise_distances(vertices).max()
# X[test_ind] += 0.04*diam*np.random.normal(size=X[test_ind].shape)

# =============================================================================
# RVGP pipeline
# =============================================================================
d = RVGP.create_data_object(X, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)
vector_field_GP = RVGP.fit(d, train_ind=train_ind, noise_variance=0.001)
f_pred_mean, _ = vector_field_GP.transform(d, test_ind)

# =============================================================================
# Plot
# =============================================================================
ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", X[train_ind])
ps_cloud.add_vector_quantity("Training vectors", d.vectors[train_ind], color=(227, 156, 28), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", X[test_ind])
ps_cloud.add_vector_quantity("Predicted vectors", f_pred_mean, color=(227, 28, 28),  enabled=True)
ps.show()
