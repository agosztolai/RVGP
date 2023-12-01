#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
import RVGP
import polyscope as ps
import numpy as np

# =============================================================================
# Parameters and data
# =============================================================================
n_eigenpairs=100
n_neighbors=10
vertices, faces = load_mesh('bunny')

# =============================================================================
# Subsample and create data object
# =============================================================================
sample_ind, _ = furthest_point_sampling(vertices, spacing=0.015)
X = vertices[sample_ind]
d = RVGP.create_data_object(X, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)

singularity = np.array([-0.08679473,  0.1146,  0.0022]) 
train_ind =  np.linalg.norm(X-singularity, axis=1) > 0.015
test_f = d.vectors[~train_ind]

vector_field_GP = RVGP.fit(d, train_ind=train_ind, noise_variance=0.001)
f_pred_mean, _ = vector_field_GP.transform(d, ~train_ind)

# =============================================================================
# Plotting
# =============================================================================
ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", X[train_ind])
ps_cloud.add_vector_quantity("Training vectors", d.vectors[train_ind], color=(227, 156, 28), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", X[~train_ind])
ps_cloud.add_vector_quantity("Predicted vectors", f_pred_mean, color=(227, 28, 28),  enabled=True)
ps.show()