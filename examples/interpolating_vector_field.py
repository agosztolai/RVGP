#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from misc import load_mesh
from RVGP.geometry import sample_from_neighbourhoods
from RVGP.kernels import ManifoldKernel
from RVGP import data, train_gp
import polyscope as ps
import numpy as np



vertices, faces = load_mesh('bunny')
dim_emb = vertices.shape[1]

<<<<<<< HEAD
# Form data object
d = data(vertices, faces, n_eigenpairs=10)
=======
n_vert = 100
if len(vertices)>n_vert:
    ind = np.random.choice(vertices.shape[0], n_vert, replace=False)
else:
    ind = np.arange(len(vertices))
>>>>>>> parent of 2111350... variational optimisation

d = data(vertices[ind], faces[ind])
d.random_vector_field()
d.smooth_vector_field(t=100)

sp_to_manifold_gp = train_gp(d.evecs_Lc.reshape(d.n, -1),
                             d.vertices,
<<<<<<< HEAD
                             n_inducing_points=10,
                             epochs=100
                             )
=======
                             noise_variance=0.001)
>>>>>>> parent of 2111350... variational optimisation

kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)

sp_to_vector_field_gp = train_gp(d.evecs_Lc.reshape(d.n, -1), 
                                 d.vectors,
                                 dim=d.vertices.shape[1],
                                 epochs=100,
                                 n_inducing_points=None,
                                 kernel=kernel
                                 )

n_test = 2
test_points = sample_from_neighbourhoods(d.evecs_Lc.reshape(d.n, -1), k=2, n=n_test)
manifold_pred_mean = sp_to_manifold_gp.predict(test_points).mean.detach().numpy()
vector_field_pred = sp_to_vector_field_gp.predict(test_points.reshape(len(test_points)*d.vertices.shape[1], -1))
vector_field_pred_mean = vector_field_pred.mean.detach().numpy().reshape(len(test_points), -1)

ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)
ps_cloud = ps.register_point_cloud("Training points", d.vertices)
ps_cloud.add_vector_quantity("Training vectors", d.vectors, color=(0.0, 0.0, 1.), enabled=True)
ps_cloud = ps.register_point_cloud("Predicted points", manifold_pred_mean)
ps_cloud.add_vector_quantity("Predicted vectors", vector_field_pred_mean, color=(1., 0., 0.), enabled=True)
ps.show()
