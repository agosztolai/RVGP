

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
from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel
from RVGP.geometry import furthest_point_sampling, sample_from_neighbourhoods, project_to_local_frame


from RVGP import data, train_gp

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
    [jnp.array([2.0, 3.0]), jnp.array([2.0, -3.0]),
     jnp.array([2.0, 2.5]), jnp.array([2.0, -2.5]),
     jnp.array([2.0, 2.0]), jnp.array([2.0, -2])],
     axis=0,
 )

steps = 2001
rollouts = system.rollout(initial_states, steps)

# subsample
#rollouts = rollouts[:,::5,:]

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


#%% convert from phase to R

def coordinates_to_phase(x, y, radius=3):
    phase_angles = np.arctan2(y, x)
    phase_angles = (phase_angles + 2 * np.pi) % (2 * np.pi)
    return phase_angles

def phase_to_coordinates(phase_angles, radius=3):
    x = radius * np.cos(phase_angles)
    y = radius * np.sin(phase_angles)
    return np.dstack((x, y))

xy = phase_to_coordinates(rollouts[:,:,0])
rollouts = np.dstack([xy , rollouts[:,:,1].reshape(rollouts.shape[0],-1,1)])


#%%

X = rollouts[:,:-1,:]
X = X.reshape([X.shape[0]*X.shape[1], X.shape[2]])
X = X.astype(jnp.float64)
X = np.array(jax.device_get(X))


# get vectors
f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.astype(jnp.float64)
f = np.array(jax.device_get(f))
f = f.reshape([f.shape[0]*f.shape[1], f.shape[2]])
#f[:, 0] = np.mod(f[:, 0] + np.pi, 2 * np.pi) - np.pi


# subsample the data
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.02) #uniform sampling of the datapoints
X = X[sample_ind]
f = f[sample_ind]
dim_emb = X.shape[1]

# create data object
d = data(X, vectors=f, n_eigenpairs=100,n_neighbors=10,)

#d.smooth_vector_field(t=100)


#%% plot the data

edgelist = [e for e in d.G.edges if e not in nx.selfloop_edges(d.G)]
G_ = d.G.copy()
G_.remove_edges_from(G_.edges())
G_.add_edges_from(edgelist)


ax = graph(G_)
#ax.quiver(X[:,0], X[:,1], f[:,0], f[:,1], color='b', scale=1)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=2)


#%% spectrum to manifold

kernel1 = gpflow.kernels.RBF()
sp_to_manifold_gp = train_gp(d.evecs_Lc.reshape(d.n, -1),
                             d.vertices,
                             dim=1,
                             epochs=2000,
                             kernel=kernel1,
                             noise_variance=0.001)


#%% spectrum to vector field

kernel2 = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                        nu=3/2, 
                        kappa=5, 
                        sigma_f=1)
#kernel = gpflow.kernels.RBF()

sp_to_vector_field_gp = train_gp(d.evecs_Lc.reshape(d.n, -1), 
                                 d.vectors,
                                 dim=d.vertices.shape[1],
                                 epochs=2000,
                                 kernel=kernel2,
                                 noise_variance=0.001)

#%% manifold to spectrum

kernel3 = gpflow.kernels.RBF()
manifold_to_sp_gp = train_gp(d.vertices,
                             d.evecs_Lc.reshape(d.n, -1),
                             dim=1,
                             epochs=2000,
                             kernel=kernel3,
                             noise_variance=0.01)

#%%

#G = manifold_graph(X)
#dim_man = 2
#gauges, Sigma = tangent_frames(X, G, dim_man, 10)
#R = connections(gauges, G, dim_man)

#%% test the prediction of manifold and vectors


#n_test = 1
#test_points = sample_from_neighbourhoods(d.evecs_Lc.reshape(d.n, -1), k=2, n=n_test)

test_points = d.evecs_Lc.reshape(d.n, -1)
X_pred, _ = sp_to_manifold_gp.predict_f(test_points)

f_pred, _ = sp_to_vector_field_gp.predict_f(test_points.reshape(len(test_points)*d.vertices.shape[1], -1))
f_pred = f_pred.numpy().reshape(len(test_points), -1)
#f_pred[:, 0] = np.mod(f_pred[:, 0] + np.pi, 2 * np.pi) - np.pi


l2_error = np.linalg.norm(f.ravel() - f_pred.ravel()) / len(f.ravel())
print("Relative l2 error is {}".format(l2_error))

ax = graph(G_)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=1)

ax = graph(G_)
ax.quiver(X[:,0], X[:,1], X[:,2], f_pred[:,0], f_pred[:,1], f_pred[:,2], color='r', length=1)

ax = graph(G_)
ax.quiver(X_pred[:,0], X_pred[:,1], X_pred[:,2], f_pred[:,0], f_pred[:,1], f_pred[:,2], color='r', length=10)



#%% project down onto plane and visualise


def project_vectors_and_positions_onto_cylinder(positions, vectors):
    n = positions.shape[0]
    projected_vectors = np.zeros((n, 2))
    projected_positions = np.zeros((n, 2))
    
    for i in range(n):
        x, y, z = positions[i]
        vector = vectors[i]

        # Compute the angular coordinate theta
        theta = np.arctan2(y, x)
        
        if theta < 0:
            theta += 2 * np.pi

        # Define the gauge directions
        gauge_theta = np.array([-y, x, 0])
        gauge_z = np.array([0, 0, 1])

        # Normalize the gauge direction in the theta direction
        gauge_theta /= np.linalg.norm(gauge_theta)

        # Define the gauge matrix
        gauge = np.column_stack((gauge_theta, gauge_z))

        # Project the vector onto the gauge directions
        projected_vector = np.dot(gauge.T, vector)

        projected_vectors[i] = projected_vector
        projected_positions[i] = [theta, z]


    return projected_positions, projected_vectors


X_proj, f_proj = project_vectors_and_positions_onto_cylinder(d.vertices, d.vectors)
X_pred_proj, f_pred_proj = project_vectors_and_positions_onto_cylinder(X_pred, f_pred)


scale = 100
width = 0.004
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.quiver(X_proj[:,0], X_proj[:,1], f_proj[:,0], f_proj[:,1],color='b')
ax.quiver(X_pred_proj[:,0], X_pred_proj[:,1], f_pred_proj[:,0], f_pred_proj[:,1], color='r')
#for i in range(rollouts.shape[0]):
#    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")

#%% test the predictions of manifold to spectrum conversion


sp_pred, _ = manifold_to_sp_gp.predict_f(d.vertices)
X_pred, _ = sp_to_manifold_gp.predict_f(sp_pred)
f_pred, _ = sp_to_vector_field_gp.predict_f(test_points.reshape(len(d.vertices)*d.vertices.shape[1], -1))
f_pred = f_pred.numpy().reshape(len(d.vertices), -1)
#f_pred[:, 0] = np.mod(f_pred[:, 0] + np.pi, 2 * np.pi) - np.pi

l2_error = np.linalg.norm(f.ravel() - f_pred.ravel()) / len(f.ravel())
print("Relative l2 error is {}".format(l2_error))

ax = graph(G_)
ax.quiver(X[:,0], X[:,1], X[:,2], f[:,0], f[:,1], f[:,2], color='g', length=2)
ax.quiver(X_pred[:,0], X_pred[:,1], X_pred[:,2], f_pred[:,0], f_pred[:,1], f_pred[:,2], color='r', length=2)


X_proj, f_proj = project_vectors_and_positions_onto_cylinder(X_pred, f_pred)
scale = 100
width = 0.004
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.quiver(X_proj[:,0], X_proj[:,1], f_proj[:,0], f_proj[:,1],)
#for i in range(rollouts.shape[0]):
#    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")


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


#%% create trjaectory


len_t = 100
h = 2 # step size

# choose a point y to find x 
idx = 0#90
#y0 = y_train[idx,:]
y0 = d.vertices[idx, :] # + np.array([0.1,-0.3])

trajectory = [y0]
vectors = []


# looping over length of trajectory
for i in range(len_t):
    
    y = trajectory[-1]  
    
    # predict the tangent space eigenvectors TU (logarithmic map)
    x_, x_var = manifold_to_sp_gp.predict_y(y.reshape(-1,1).T)
    
    # predict the manifold point from TU (exponential map)
    y_pred, _ = sp_to_manifold_gp.predict_f(x_) # should be similar to y
    y_pred = y_pred.numpy()
    
    # predict the vector from the TU
    x_ = x_.numpy()
    f_pred_, f_var = sp_to_vector_field_gp.predict_f(x_.reshape(1*d.vertices.shape[1], -1))
    f_pred_ = f_pred_.numpy().reshape(1, -1)
   
    # perform euler iteration    
    y1 = y + h * f_pred_    
    
    # append to trajectory
    trajectory.append(y1.squeeze())
    vectors.append(f_pred_)



# create plot 3d
ax = graph(G_)
#ax.quiver(X[:,0], X[:,1], f[:,0], f[:,1], color='g', scale=0.3)
ax.quiver(X[:,0], X[:,1],X[:,2],  f[:,0], f[:,1], f[:,2],  color='g', length=1)

for i in range(0,len_t,3):
    y = trajectory[i]
    f_traj = vectors[i]
    #ax.quiver(y[0], y[1],  f_traj[0,0], f_traj[0,1], color='r', scale=0.3)
    ax.quiver(y[0], y[1], y[2], f_traj[0,0], f_traj[0,1],  f_traj[0,2], color='r', length=5)



# create plot 2d
X_proj, f_proj = project_vectors_and_positions_onto_cylinder(X_pred, f_pred)
scale = 100
width = 0.004
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.quiver(X_proj[:,0], X_proj[:,1], f_proj[:,0], f_proj[:,1],)
#for i in range(rollouts.shape[0]):
#    plt.scatter(rollouts[i, :, 0], rollouts[i, :, 1], s=1)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.gca().set_aspect("equal")


X_proj_traj, f_proj_traj = project_vectors_and_positions_onto_cylinder(np.vstack(trajectory)[:-1,:], np.vstack(vectors))
X_proj_traj[:, 0] = np.mod(X_proj_traj[:, 0] , 2 * np.pi) 

for i in range(0,len_t,3):
    y = X_proj_traj[i]
    f_traj =  f_proj_traj[i]
    #ax.quiver(y[0], y[1],  f_traj[0,0], f_traj[0,1], color='r', scale=0.3)
    ax.quiver(y[0], y[1], f_traj[0], f_traj[1],  color='r')


