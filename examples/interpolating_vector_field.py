#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gpflow

from ptu_dijkstra import connections, tangent_frames

from misc import sample_spherical
from RVGP.geometry import (furthest_point_sampling, 
                           manifold_graph, 
                           compute_laplacian,
                           compute_connection_laplacian,
                           compute_spectrum,
                           sample_from_convex_hull,
                           project_to_manifold, 
                           project_to_local_frame,
                           node_eigencoords
                           )
from RVGP.smoothing import vector_diffusion
from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel

np.random.seed(0)

# =============================================================================
# Define manifold points
# =============================================================================
X = sample_spherical(300)
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.1) #uniform sampling of the datapoints
X = X[sample_ind]
dim_emb = X.shape[1]

# =============================================================================
# Fit graph, tangent frames and connections
# =============================================================================
G = manifold_graph(X)
dim_man = 2
gauges, Sigma = tangent_frames(X, G, dim_man, 10)
R = connections(gauges, G, dim_man)

# =============================================================================
# Eigendecomposition of connection Laplacian
# =============================================================================
L = compute_laplacian(G)
Lc = compute_connection_laplacian(G, R)

n_eigenpairs = 50
evals_Lc, evecs_Lc = compute_spectrum(Lc, n_eigenpairs) # U\Lambda U^T

#rather than U, take TU, where T is the local gauge
evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man,n_eigenpairs)
evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)

# =============================================================================
# Define vector field over manifold and smooth it using vector diffusion
# =============================================================================
f = np.random.uniform(size=(len(X), 3))-.5
f = project_to_manifold(f, gauges[...,:2])
f /= np.linalg.norm(f, axis=1, keepdims=True)

ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='b', length=0.3)

t=100
f = project_to_local_frame(f, gauges)
f = vector_diffusion(f, t, L=L, Lc=Lc, method="matrix_exp")
f = project_to_local_frame(f, gauges, reverse=True)

ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)
ax.axis('equal')

# =============================================================================
# Custom kernel
# =============================================================================
kernel = ManifoldKernel((evecs_Lc, evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)

# =============================================================================
# Train GP for manifold
# =============================================================================
x_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)
y_train = X
manifold_GP = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP.training_loss, manifold_GP.trainable_variables)

# =============================================================================
# Train GP for vector field over manifold
# =============================================================================
node_train = np.random.choice(len(X), size=int(len(X)*0.8), replace=False)[:,None]
node_test = np.arange(len(X), dtype=int).reshape(-1, 1)
x_train = node_eigencoords(node_train, evecs_Lc, dim_emb)
f_train = f[node_train.flatten()].reshape(-1, 1)
x_test = node_eigencoords(node_test, evecs_Lc, dim_emb)
vector_field_GP = gpflow.models.GPR((x_train, f_train), kernel=kernel, noise_variance=0.001)

# optimize_GPR(vector_field_GP, 10000) #this is an alternative using gradient descent
# gpflow.utilities.print_summary(vector_field_GP)    
opt = gpflow.optimizers.Scipy()
opt.minimize(vector_field_GP.training_loss, vector_field_GP.trainable_variables)

# =============================================================================
# Test performance
# =============================================================================
f_pred_mean, _ = vector_field_GP.predict_f(x_test) 
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

l2_error = np.linalg.norm(f.ravel() - f_pred_mean.ravel()) / len(f.ravel())
print("Relative l2 error is {}".format(l2_error))

ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)
ax.quiver(X[:,0], X[:,1], X[:,2], f_pred_mean[:,0], f_pred_mean[:,1], f_pred_mean[:,2], color='r', length=0.3)
ax.axis('equal')

# =============================================================================
# Make new predictions at interpolated points
# =============================================================================
x_test = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)
x_test = sample_from_convex_hull(x_test, 60, k=3)
y_pred_mean, _ = manifold_GP.predict_f(x_test)
x_test = x_test.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(x_test)
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)
ax.quiver(y_pred_mean[:,0], y_pred_mean[:,1], y_pred_mean[:,2], f_pred_mean[:,0], f_pred_mean[:,1], f_pred_mean[:,2], color='r', length=0.3)
<<<<<<< Updated upstream
ax.axis('equal')
=======
ax.axis('equal')


#%%
# =============================================================================
# Train GP for predicting spectral coordinates given some real point coordinates
# =============================================================================


x_train = X
y_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)

manifold_GP_inv = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP_inv.training_loss, manifold_GP_inv.trainable_variables)


y_pred_mean, _ = manifold_GP_inv.predict_f(x_train)
y_pred_mean = y_pred_mean.numpy()
# x_test = x_test.T
# np.random.shuffle(x_test)
# x_test = x_test.T

y_pred_mean = y_pred_mean.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(y_pred_mean)
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)
ax.quiver(X[:,0], X[:,1], X[:,2], f_pred_mean[:,0], f_pred_mean[:,1], f_pred_mean[:,2], color='r', length=0.3)
ax.axis('equal')


#%%

# =============================================================================
# Find trajectory in the vector field
# =============================================================================

def project_point_onto_tangent_space(x, u, v, y):
    # Ensure inputs are numpy arrays
    x, u, v, y = np.array(x), np.array(u), np.array(v), np.array(y)

    # Step 1: Find the displacement vector from x to y
    displacement = y - x

    # Step 2: Calculate the dot products with the basis vectors
    dot_u = np.dot(displacement, u)
    dot_v = np.dot(displacement, v)

    # Step 3: Construct the projection of y onto the tangent space
    projection = x + dot_u * u + dot_v * v

    return projection

# from RVGP.trajectory import project_to_eigencoords
# from scipy.optimize import minimize

# parameters
len_t = 50
h = 0.1 # step size
n_sample_points = 10

# choose a point y to find x 
idx = 10#87#90
#y0 = y_train[idx,:]
y0 = x_train[idx, :]

trajectory = [y0]
vectors = []

# import scipy as sc

# looping over length of trajectory
for i in range(len_t):
    
    y = trajectory[-1]  
    
    # predict the eigenvectors: TU
    x_, x_var = manifold_GP_inv.predict_y(y.reshape(-1,1).T)
    
    # predict the manifold point from TU
    y_pred, _ = manifold_GP.predict_f(x_) # should be similar to y
    y_pred = y_pred.numpy()
    
    # predict the vector from the TU
    f_pred, f_var = vector_field_GP.predict_f(x_.numpy().reshape(3,50))
    f_pred = f_pred.numpy().reshape(-1,dim_emb)
   
    # perform euler iteration    
    y1 = y_pred + h * f_pred
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    vectors.append(f_pred)
    
    
    # dist = sc.spatial.distance.cdist(y.reshape(-1,1).T,X)
    # nearest_point = dist.argmin()

    # n_neighbours = 5
    # neighbourhood = np.argsort(dist)[:,:n_neighbours].reshape(-1)

    # # project down onto the local tangent space
    # neighbour_gauges = np.squeeze(gauges[neighbourhood,:,:])

    # projections = []    
    # for i in range(n_neighbours):        
    #     proj = project_point_onto_tangent_space(X[neighbourhood[i],:], neighbour_gauges[i,:,0], neighbour_gauges[i,:,1], y)
    #     projections.append(proj)
    
    
    # projections = np.stack(projections)    
    
    # # finding average projection down onto local tangent spaces
    # y_proj = np.mean(projections,0)
    

    




# create plot
ax = graph(G)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=0.3)

for i in range(0,len_t,3):
    y = trajectory[i]
    f_traj = vectors[i]
    ax.quiver(y[0], y[1], y[2], f_traj[0,0], f_traj[0,1], f_traj[0,2], color='r', length=0.3)
    
    
    
#%% visualise variance on the sphere





>>>>>>> Stashed changes
