#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from dataclasses import dataclass

from ptu_dijkstra import connections, tangent_frames
from RVGP.geometry import (manifold_graph, 
                           compute_laplacian,
                           compute_connection_laplacian,
                           compute_spectrum,
                           express_in_local_frame,
                           project_to_manifold,
                           manifold_dimension
                           )

from RVGP.smoothing import vector_diffusion

@dataclass
class data:
    def __init__(self, 
                 vertices,
                 vectors=None,
                 dim_man=2, 
                 n_neighbors=10,
                 frac_geodesic_neighbours=1.5,
                 explained_variance=0.8,
                 n_eigenpairs=None):
        
        print('Fit graph')
        G = manifold_graph(vertices, n_neighbors=n_neighbors)
        
        print('Fit tangent spaces')
        gauges, Sigma = tangent_frames(vertices, G, vertices.shape[1], n_neighbors*frac_geodesic_neighbours)
        
        dim_man = manifold_dimension(Sigma, frac_explained=explained_variance)
        gauges = gauges[:,:,:dim_man]
        print('Predicted manifold dimension is {}'.format(dim_man))
        
        R = connections(gauges, G, dim_man) 
        print('Fit connections')
        
        print('Compute Laplacians')
        L = compute_laplacian(G)
        Lc = compute_connection_laplacian(G, R)
        
        print('Compute eigendecompositions')
        evals_L, evecs_L = compute_spectrum(L, n_eigenpairs)
        evals_L, evecs_L = evals_L.numpy(), evecs_L.numpy()
        
        evals_Lc, evecs_Lc = compute_spectrum(Lc, n_eigenpairs) # U\Lambda U^T
        #rather than U, take TU, where T is the local gauge
        if n_eigenpairs is None:
            n_eigenpairs = evecs_Lc.shape[1]
        evals_Lc = evals_Lc.numpy()
        evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man, n_eigenpairs)
        evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
        evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)
        
        (
            self.vertices,
            self.n,
            self.dim_man,
            self.G, 
            self.gauges, 
            self.R, 
            self.L, 
            self.Lc,
            self.evals_L,
            self.evecs_L,
            self.evals_Lc,
            self.evecs_Lc,  
            self.vectors
        ) = (
                vertices,
                vertices.shape[0],
                dim_man,
                G, 
                gauges, 
                R, 
                L, 
                Lc,
                evals_L,
                evecs_L,
                evals_Lc,
                evecs_Lc,
                vectors
            )
        
    def random_vector_field(self, seed=0):
        """Generate random vector field over manifold"""
        
        np.random.seed(seed)
        
        vectors = np.random.uniform(low=-0.5,
                                    high=0.5,
                                    size=(len(self.vertices), self.vertices.shape[1])
                                    )
        vectors = project_to_manifold(vectors, self.gauges[...,:self.dim_man])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        
        self.vectors = vectors
            
    def smooth_vector_field(self, t=100):
        
        if hasattr(self, 'vectors'):
    
            """Smooth vector field over manifold"""
            vectors = express_in_local_frame(self.vectors, self.gauges)
            vectors = vector_diffusion(vectors, 
                                       t, 
                                       L=self.L, 
                                       Lc=self.Lc, 
                                       method="matrix_exp")
            vectors = express_in_local_frame(vectors, self.gauges, reverse=True)
        
            self.vectors = vectors
        else:
            print('No vectors found. Nothing to smooth.')