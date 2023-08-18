

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
from RVGP.plotting import graph
from RVGP.kernels import ManifoldKernel


import torch
from rnn.modules import LatentLowRankRNN, train
from rnn.RNN_helpers import plot_field_noscalings, plot_trajectories



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
rollouts = rollouts[:,::10,:]

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

#%% convert phase to x,y

def phase_to_coordinates(phase_angles, radius=1):
    x = radius * np.cos(phase_angles)
    y = radius * np.sin(phase_angles)
    return np.dstack((x, y))

xy = phase_to_coordinates(rollouts[:,:,0])
rollouts = np.dstack([xy , rollouts[:,:,1].reshape(2,-1,1)])

#%%

# trajectory
X = rollouts[:,:-1,:]
#X = X.reshape([X.shape[0]*X.shape[1], X.shape[2]])
X = X.astype(jnp.float32)
X = np.array(jax.device_get(X))
X = torch.from_numpy(X)

# vectors
f = rollouts[:, 1:, :] - rollouts[:, :-1, :]
f = f.astype(jnp.float64)
f = np.array(jax.device_get(f))
f = f.reshape([f.shape[0]*f.shape[1], f.shape[2]])
# f[:, 0] = np.mod(f[:, 0] + np.pi, 2 * np.pi) - np.pi


# what is this?
label = torch.rand(X.shape[0], X.shape[1], 2)


#%%

X_ = X.reshape([X.shape[0]*X.shape[1], X.shape[2]])
sample_ind, _ = furthest_point_sampling(X_, stop_crit=0.02) #uniform sampling of the datapoints
X_ = X_[sample_ind]
f = f[sample_ind]
dim_emb = X_.shape[1]

G = manifold_graph(X_)
dim_man = 2
#gauges, Sigma = tangent_frames(X, G, dim_man, 10)
#R = connections(gauges, G, dim_man)



#%% train RNN

noise_std = 5e-2
alpha = 0.2
rank = 10

net = LatentLowRankRNN(3, [16,32,512], rank=rank, alpha=alpha, noise_std=noise_std)
#train(net, X, label, epochs=150, batch_size=32, lr=0.01)
train(net, X, epochs=150, batch_size=32, lr=0.01)

#%%
x_ = X[:,[0],:].to(device=net.m.device)
x, _, _ = net(x_, seq_len=X.shape[1])
x = x.detach().cpu().numpy()

edgelist = [e for e in G.edges if e not in nx.selfloop_edges(G)]
G_ = G.copy()
G_.remove_edges_from(G_.edges())
G_.add_edges_from(edgelist)

ax = graph(G_)
ax.quiver(X_[:,0], X_[:,1],X_[:,2], f[:,0], f[:,1],f[:,2], color='b', length=1)


#%%

x_ = X[:,[0],:].to(device=net.m.device)
x, _, _ = net(x_, seq_len=X.shape[1])
x = x.detach().cpu().numpy()

edgelist = [e for e in G.edges if e not in nx.selfloop_edges(G)]
G_ = G.copy()
G_.remove_edges_from(G_.edges())
G_.add_edges_from(edgelist)

ax = graph(G_)
ax.quiver(X_[:,0], X_[:,1], f[:,0], f[:,1], color='b', scale=3)
for i in range(x.shape[0]):
    ax.plot(x[i,:,0], x[i,:,1], color='r')
ax.axis('equal')




#%%


ax, _ = plot_field_noscalings(net, input=None, xmin=-5, xmax=5, ymin=-5, ymax=5, add_fixed_points=False)

_, _, z = net(X[0][0:2].unsqueeze(1).to(device=net.m.device))
plot_trajectories(net, z.detach().cpu().numpy() , ax, n_traj=2, style="-", c="C0")

