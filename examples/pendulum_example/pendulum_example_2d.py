

import jax.config
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from dynamical_systems import FrictionPendulumSystem
import networkx as nx

from utils import (
    GlobalRNG,
    mesh_to_polyscope,
    project_to_3d,
    cylinder_m_to_3d,
    cylinder_projection_matrix_to_3d,
    normalise_scaled_kernel,
    plot_scalar_field,
    plot_vector_field,
    plot_covariances,
    plot_mean_cov,
    plot_inference,
    plot_2d_sparse_gp,
    circle_distance,
)

from dynamical_systems_utils import (
    reverse_eular_integrate_rollouts,   
    )


import gpflow

from ptu_dijkstra import connections, tangent_frames

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


set_matplotlib_formats("svg")
rng = GlobalRNG()


# %% Set up the phase space grid
n_points = 15
positions = jnp.linspace(0, 2 * jnp.pi, n_points)
momentums = jnp.linspace(-3, 3, n_points)
positions, momentums = jnp.meshgrid(positions, momentums)
positions = positions.flatten()[:, np.newaxis]
momentums = momentums.flatten()[:, np.newaxis]
phase_space = m = jnp.concatenate([positions, momentums], axis=-1)

# %% Initialise a pendulum system with friction and get the dynamics field
system = FrictionPendulumSystem(mass=0.1, length=2.0, friction=0.03)
scale = 300
hgf = system.hamiltonian_gradient_field(positions, momentums)
ncf = system.non_conservative_field(positions, momentums)
dgf = system.dynamics_gradient_field(positions, momentums)
h = system.hamiltonian(positions, momentums)

# %% Plot the dynamice field
scale = 100
width = 0.004
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
plot_vector_field(phase_space, dgf, color="black", ax=ax, scale=scale, width=width)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")
#plt.savefig("../figures/dynamics_statespace.pdf", bbox_inches="tight")

# %% rollout 2 trajectories to train on
initial_states = jnp.stack(
    [jnp.array([2.0, 3.0]), jnp.array([2.0, -3.0])],
    axis=0,
)
steps = 2001
rollouts = system.rollout(initial_states, steps)

# %% Plot the trainign data
scale = 100
width = 0.004
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
plot_vector_field(phase_space, dgf, ax=ax, color="black", scale=scale)
for i in range(rollouts.shape[0]):
    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")
#plt.savefig("../figures/dynamics_training_data.pdf", bbox_inches="tight")

# %% compute reverse integrated trajectory data from the rollouts

scale = 400
m_cond, v_cond = reverse_eular_integrate_rollouts(
    rollouts, system, thinning=5, chuck_factor=4
)
m_cond_, v_cond_ = reverse_eular_integrate_rollouts(
    rollouts, system, thinning=5, chuck_factor=4, estimate_momentum=True
)


#%%

X = rollouts[:,:-1,:]
X = X.reshape([X.shape[0]*X.shape[1], X.shape[2]])
X = X.astype(jnp.float64)
X = np.array(jax.device_get(X))


#%% get vectors

f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.astype(jnp.float64)
f = np.array(jax.device_get(f))
f = f.reshape([f.shape[0]*f.shape[1], f.shape[2]])
f[:, 0] = np.mod(f[:, 0] + np.pi, 2 * np.pi) - np.pi


#f = f.reshape([f.shape[0]*f.shape[1], f.shape[2]])


#%% subsample the data

sample_ind, _ = furthest_point_sampling(X, stop_crit=0.02) #uniform sampling of the datapoints
X = X[sample_ind]
f = f[sample_ind]
dim_emb = X.shape[1]

G = manifold_graph(X)
dim_man = 2
gauges, Sigma = tangent_frames(X, G, dim_man, 10)
R = connections(gauges, G, dim_man)

#%%
# get spectral features of manifold
L = compute_laplacian(G)
Lc = compute_connection_laplacian(G, R)

n_eigenpairs = 50
evals_Lc, evecs_Lc = compute_spectrum(Lc, n_eigenpairs) # U\Lambda U^T

#rather than U, take TU, where T is the local gauge
evecs_Lc = evecs_Lc.numpy().reshape(-1, dim_man,n_eigenpairs)
evecs_Lc = np.einsum("bij,bjk->bik", gauges, evecs_Lc)
evecs_Lc = evecs_Lc.reshape(-1, n_eigenpairs)

#%%

edgelist = [e for e in G.edges if e not in nx.selfloop_edges(G)]
G_ = G.copy()
G_.remove_edges_from(G_.edges())
G_.add_edges_from(edgelist)


ax = graph(G_)
ax.quiver(X[:,0], X[:,1], f[:,0], f[:,1], color='b', scale=1)
ax.axis('equal')


# %% Create a manifold kernel for the GP

kernel = ManifoldKernel((evecs_Lc, evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)

# kernel = ManifoldKernel((evecs_Lc, evals_Lc), 
#                         nu=3/2, 
#                         kappa=5, 
#                         sigma_f=1)


# %% Plot the kernel


# k = kernel.matrix(kernel_params, phase_space, phase_space)

# i = int(n_points ** 2 / 2)
# plt.contourf(
#     phase_space[:, 0].reshape(n_points, n_points),
#     phase_space[:, 1].reshape(n_points, n_points),
#     k[:, i, 0, 0].reshape(n_points, n_points),
#     50,
# )
# plt.scatter(phase_space[i, 0], phase_space[i, 1])
# plt.colorbar()

#%% train manifold GP

x_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs) # evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)
y_train = X

manifold_GP = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP.training_loss, manifold_GP.trainable_variables)

#%% train vector field GP

node_train = np.random.choice(len(X), size=int(len(X)*0.8), replace=False)[:,None]
node_test = np.arange(len(X), dtype=int).reshape(-1, 1)


x_train_f = node_eigencoords(node_train, evecs_Lc, dim_emb)
f_train = f[node_train.flatten()].reshape(-1, 1)
x_test = node_eigencoords(node_test, evecs_Lc, dim_emb)
vector_field_GP = gpflow.models.GPR((x_train_f, f_train), kernel=kernel, noise_variance=0.001)

# optimize_GPR(vector_field_GP, 10000) #this is an alternative using gradient descent
# gpflow.utilities.print_summary(vector_field_GP)    
opt = gpflow.optimizers.Scipy()
opt.minimize(vector_field_GP.training_loss, vector_field_GP.trainable_variables)

#%% test accuracy
f_pred_mean, _ = vector_field_GP.predict_f(x_test) 
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

l2_error = np.linalg.norm(f.ravel() - f_pred_mean.ravel()) / len(f.ravel())
print("Relative l2 error is {}".format(l2_error))

ax = graph(G_)
ax.quiver(X[:,0], X[:,1], f[:,0], f[:,1], color='g', scale=0.3)
#ax.quiver(X[:,0], X[:,1], f_pred_mean[:,0], f_pred_mean[:,1], color='r',)
ax.axis('equal')

#%%

x_train = X
y_train = evecs_Lc.reshape(-1, dim_emb*n_eigenpairs)

manifold_GP_inv = gpflow.models.GPR((x_train, y_train), kernel=gpflow.kernels.RBF(), noise_variance=0.01)

opt = gpflow.optimizers.Scipy()
opt.minimize(manifold_GP_inv.training_loss, manifold_GP_inv.trainable_variables)


y_pred_mean, _ = manifold_GP_inv.predict_f(x_train)
y_pred_mean = y_pred_mean.numpy()


y_pred_mean = y_pred_mean.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(y_pred_mean)
f_pred_mean = f_pred_mean.numpy().reshape(-1,dim_emb)

ax = graph(G_)
ax.quiver(X[:,0], X[:,1],  f[:,0], f[:,1],  color='g', scale=0.3)
ax.quiver(X[:,0], X[:,1],  f_pred_mean[:,0], f_pred_mean[:,1], color='r', scale=0.3)
ax.axis('equal')

#%% create trjaectory


len_t = 200
h = 5 # step size
n_sample_points = 10

# choose a point y to find x 
idx = 0#90
#y0 = y_train[idx,:]
y0 = x_train[idx, :] # + np.array([0.1,-0.3])

trajectory = [y0]
vectors = []


# looping over length of trajectory
for i in range(len_t):
    
    y = trajectory[-1]  
    
    # predict the tangent space eigenvectors TU (logarithmic map)
    x_, x_var = manifold_GP_inv.predict_y(y.reshape(-1,1).T)
    
    # predict the manifold point from TU (exponential map)
    y_pred, _ = manifold_GP.predict_f(x_) # should be similar to y
    y_pred = y_pred.numpy()
    
    # predict the vector from the TU
    f_pred, f_var = vector_field_GP.predict_f(x_.numpy().reshape(2,50))
    f_pred = f_pred.numpy().reshape(-1,dim_emb)
   
    # perform euler iteration    
    y1 = y + h * f_pred    
    y1[:, 0] = np.mod(y1[:, 0] , 2 * np.pi) 
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    vectors.append(f_pred)


# create plot
ax = graph(G_)
ax.quiver(X[:,0], X[:,1], f[:,0], f[:,1], color='g', scale=0.3)

for i in range(0,len_t,3):
    y = trajectory[i]
    f_traj = vectors[i]
    ax.quiver(y[0], y[1],  f_traj[0,0], f_traj[0,1], color='r', scale=0.3)
    
    