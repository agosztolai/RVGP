#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from dataclasses import dataclass

from ptu_dijkstra import connections, tangent_frames
from RVGP.geometry import (manifold_graph, 
                           compute_laplacian,
                           compute_connection_laplacian,
                           compute_spectrum,
                           project_to_local_frame,
                           project_to_manifold
                           )

from RVGP.smoothing import vector_diffusion

@dataclass
class data:
    def __init__(self, 
                 vertices,
                 faces,
                 dim_man=2, 
                 n_eigenpairs = 50):
        
        print('Fit graph')
        G = manifold_graph(vertices)
        print('Fit tangent spaces and connections')
        gauges, Sigma = tangent_frames(vertices, G, dim_man, 10)
        R = connections(gauges, G, dim_man)
        
        print('Compute Laplacians')
        L = compute_laplacian(G)
        Lc = compute_connection_laplacian(G, R)
        
        print('Compute eigendecompositions')
        evals_Lc, evecs_Lc = compute_spectrum(Lc, n_eigenpairs) # U\Lambda U^T
        #rather than U, take TU, where T is the local gauge
        evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man, n_eigenpairs)
        evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
        evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)
        
        (
            self.vertices,
            self.faces,
            self.n,
            self.dim_man,
            self.G, 
            self.gauges, 
            self.R, 
            self.L, 
            self.Lc,
            self.evals_Lc,
            self.evecs_Lc   
        ) = (
                vertices,
                faces,
                vertices.shape[0],
                dim_man,
                G, 
                gauges, 
                R, 
                L, 
                Lc,
                evals_Lc,
                evecs_Lc
            )
        
    def random_vector_field(self, seed=0):
        """Generate random vector field over manifold"""
        
        np.random.seed(0)
        
        vectors = np.random.uniform(size=(len(self.vertices), 3))-.5
        vectors = project_to_manifold(vectors, self.gauges[...,:2])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        
        self.vectors = vectors
            
    def smooth_vector_field(self, t=100):
        
        if hasattr(self, 'vectors'):
    
            """Smooth vector field over manifold"""
            vectors = project_to_local_frame(self.vectors, self.gauges)
            vectors = vector_diffusion(vectors, t, L=self.L, Lc=self.Lc, method="matrix_exp")
            vectors = project_to_local_frame(vectors, self.gauges, reverse=True)
        
            self.vectors = vectors
        else:
            print('No vectors found. Nothing to smooth.')