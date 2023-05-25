#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:00:53 2023

@author: gosztola
"""

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
import numpy as np
from numpy.linalg import inv

# import pickle
# import gpflow
import tensorflow as tf
# import tensorflow_probability as tfp
from scipy import sparse

# from tqdm import trange


from sklearn.neighbors import kneighbors_graph

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    
# def plot_gp_2D(mu, X_train, Y_train, title, i):
#     # ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
#     # ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
#     ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
#     ax.set_title(title)
    
    
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def create_axis(*args, fig=None):
    """Create axis."""
    dim = args[0]
    if len(args) > 1:
        args = [args[i] for i in range(1, len(args))]
    else:
        args = (1, 1, 1)

    if fig is None:
        fig = plt.figure()

    if dim == 2:
        ax = fig.add_subplot(*args)
    elif dim == 3:
        ax = fig.add_subplot(*args, projection="3d")

    return fig, ax


def set_axes(ax, lims=None, padding=0.1, axes_visible=True):
    """Set axes."""
    if lims is not None:
        xlim = lims[0]
        ylim = lims[1]
        pad = padding * (xlim[1] - xlim[0])

        ax.set_xlim([xlim[0] - pad, xlim[1] + pad])
        ax.set_ylim([ylim[0] - pad, ylim[1] + pad])
        if ax.name == "3d":
            zlim = lims[2]
            ax.set_zlim([zlim[0] - pad, zlim[1] + pad])

    if not axes_visible:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if ax.name == "3d":
            ax.set_zticklabels([])
        ax.axis("off")


def graph(
    G,
    labels="b",
    edge_width=1,
    edge_alpha=1.0,
    node_size=20,
    layout=None,
    ax=None,
    axes_visible=True,
):
    """Plot scalar values on graph nodes embedded in 2D or 3D."""

    G = nx.convert_node_labels_to_integers(G)
    pos = list(nx.get_node_attributes(G, "pos").values())

    if not pos:
        if layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)

    dim = len(pos[0])
    assert dim in (2, 3), "Dimension must be 2 or 3."

    if ax is None:
        _, ax = create_axis(dim)

    if dim == 2:
        if labels is not None:
            nx.draw_networkx_nodes(
                G, pos=pos, node_size=node_size, node_color=labels, alpha=0.8, ax=ax
            )

        nx.draw_networkx_edges(G, pos=pos, width=edge_width, alpha=edge_alpha, ax=ax)

    elif dim == 3:
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        if labels is not None:
            ax.scatter(*node_xyz.T, s=node_size, c=labels, ec="w")

        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray", alpha=edge_alpha, linewidth=edge_width)

    set_axes(ax, axes_visible=axes_visible)

    return ax


def compute_laplacian(G, normalization=None):

    laplacian = sparse.csr_matrix(nx.laplacian_matrix(G), dtype=np.float64)

    if normalization == "rw":
        deg = np.array([G.degree[i] for i in G.nodes])
        laplacian /= deg
        laplacian = sparse.csr_matrix(laplacian, dtype=np.float64)
    
    return laplacian


def compute_spectrum(laplacian, n_eigenpairs=None, dtype=tf.float64):
    
    if n_eigenpairs is None:
        n_eigenpairs = laplacian.shape[0]
    if n_eigenpairs >= laplacian.shape[0]:
        print("Number of features is greater than number of vertices. Number of features will be reduced.")
        n_eigenpairs = laplacian.shape[0]

    evals, evecs = tf.linalg.eigh(laplacian.toarray())
    evals = evals[:n_eigenpairs]
    evecs = evecs[:, :n_eigenpairs]/np.sqrt(len(evecs))

    evals = tf.convert_to_tensor(evals, dtype=dtype)
    evecs = tf.convert_to_tensor(evecs, dtype)
    
    return evals, evecs


import scipy

def sample_from_convex_hull(points, num_samples, k=5):
    
    tree = scipy.spatial.KDTree(points)
    
    if num_samples > len(points):
        num_samples = len(points)
    
    sample_points = np.random.choice(len(points), size=num_samples, replace=False)
    sample_points = points[sample_points]

    # Generate samples
    samples = []
    for current_point in sample_points:
        _, nn_ind = tree.query(current_point, k=k, p=2)
        nn_hull = points[nn_ind]
        
        barycentric_coords = np.random.uniform(size=nn_hull.shape[0])
        barycentric_coords /= np.sum(barycentric_coords)
        
        current_point = np.sum(nn_hull.T * barycentric_coords, axis=1)

        samples.append(current_point)

    return np.array(samples)


from sklearn.metrics import pairwise_distances

def manifold_graph(X, typ = 'knn', n_neighbors=5):
    if typ == 'knn':
        A = kneighbors_graph(X, n_neighbors, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)
        A += sparse.eye(A.shape[0])
        G = nx.from_scipy_sparse_array(A)
        
    elif typ == 'affinity':
        pairwise_distances_sphere = pairwise_distances(X)
        sigma = 0.1  # Control the width of the Gaussian kernel
        A = np.exp(-pairwise_distances_sphere ** 2 / (2 * sigma ** 2))
        G = nx.from_numpy_array(A)
        
    node_attribute = {i: X[i] for i in G.nodes}
    nx.set_node_attributes(G, node_attribute, "pos")

    return G, A


def furthest_point_sampling(x, N=None, stop_crit=0.1, start_idx=0):
    """A greedy O(N^2) algorithm to do furthest points sampling

    Args:
        x (nxdim matrix): input data
        N (int): number of sampled points
        stop_crit: when reaching this fraction of the total manifold diameter, we stop sampling
        start_idx: index of starting node

    Returns:
        perm: node indices of the N sampled points
        lambdas: list of distances of furthest points
    """
    if stop_crit == 0.0:
        return np.arange(len(x)), None

    D = pairwise_distances(x)
    n = D.shape[0] if N is None else N
    diam = D.max()

    start_idx = 5

    perm = np.zeros(n, dtype=np.int32)
    perm[0] = start_idx
    lambdas = np.zeros(n)
    ds = D[start_idx, :]
    for i in range(1, n):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])

        if N is None:
            if lambdas[i] / diam < stop_crit:
                perm = perm[:i]
                lambdas = lambdas[:i]
                break

    return perm, lambdas


# def optimize_GPR(model, train_steps):
#     adam_opt = tf.optimizers.Adam()
#     adam_opt.minimize(loss=model.training_loss, var_list=model.trainable_variables)

#     t = trange(train_steps - 1)
#     for step in t:
#         adam_opt.minimize(model.training_loss, var_list=model.trainable_variables)
#         if step % 200 == 0:
#             t.set_postfix({'likelihood': -model.training_loss().numpy()})


def compute_connection_laplacian(G, R, normalization=None):
    r"""Connection Laplacian

    Args:
        data: Pytorch geometric data object.
        R (nxnxdxd): Connection matrices between all pairs of nodes. Default is None,
            in case of a global coordinate system.
        normalization: None, 'sym', 'rw'
                 1. None: No normalization
                 :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

                 2. "sym"`: Symmetric normalization
                 :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
                 \mathbf{D}^{-1/2}`

                 3. "rw"`: Random-walk normalization
                 :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

    Returns:
        ndxnd normalised connection Laplacian matrix.
    """
    n = len(G)
    dim = R.shape[0] // n
    
    # unnormalised (combinatorial) laplacian, to be normalised later
    laplacian = compute_laplacian(G)    
    
    # rearrange into block form (kron(L, ones(d,d)))
    # edge_index = utils.expand_edge_index(L.indices(), dim=d)
    # L = torch.sparse_coo_tensor(edge_index, L.values().repeat_interleave(d * d))
    laplacian = sparse.kron(laplacian, np.ones([dim, dim]))
    
    # unnormalised connection laplacian
    # Lc(i,j) = L(i,j)*R(i,j) if (i,j)=\in E else 0
    Lc = laplacian.multiply(R)

    # normalize
    if normalization == "rw":
        deg = np.array(list(dict(G.degree()).values()))
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float("inf")] = 0
        deg_inv = deg_inv.repeat(dim, axis=0)
        Lc = sparse.diags(deg_inv, 0, format='csr') @ Lc

    elif normalization == "sym":
        raise NotImplementedError

    return Lc


def scalar_diffusion(x, t, method="matrix_exp", par=None):
    """Scalar diffusion."""
    if len(x.shape) == 1:
        x = x.unsqueeze(1)

    if method == "matrix_exp":
        par = par.todense()
        return scipy.linalg.expm(-t * par) @ x

    if method == "spectral":
        assert (
            isinstance(par, (list, tuple)) and len(par) == 2
        ), "For spectral method, par must be a tuple of \
            eigenvalues, eigenvectors!"
        evals, evecs = par

        # Transform to spectral
        x_spec = np.mm(evecs.T, x)

        # Diffuse
        diffusion_coefs = np.exp(-evals[...,None] * t)
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex
        return evecs.mm(x_diffuse_spec)

    raise NotImplementedError


def vector_diffusion(x, t, Lc, L=None, method="spectral", normalise=True):
    """Vector diffusion."""
    n, d = x.shape[0], x.shape[1]

    if method == "spectral":
        assert len(Lc) == 2, "Lc must be a tuple of eigenvalues, eigenvectors!"
        nd = Lc[0].shape[0]
    else:
        nd = Lc.shape[0]

    assert (
        n * d % nd
    ) == 0, "Data dimension must be an integer multiple of the dimensions \
         of the connection Laplacian!"

    # vector diffusion with connection Laplacian
    out = x.reshape(nd, -1)
    out = scalar_diffusion(out, t, method, Lc)
    out = out.reshape(x.shape)

    if normalise:
        assert L is not None, 'Need Laplacian for normalised diffusion!'
        x_abs = np.linalg.norm(x, axis=-1, keepdims=True)
        out_abs = scalar_diffusion(x_abs, t, method, L)
        ind = scalar_diffusion(np.ones([x.shape[0],1]), t, method, L)
        out = out*out_abs/(ind*np.linalg.norm(out, axis=-1, keepdims=True))

    return out


def project_to_manifold(x, gauges):
    coeffs = np.einsum("bij,bi->bj", gauges, x)
    return np.einsum("bj,bij->bi", coeffs, gauges)


def project_to_local_frame(x, gauges, reverse=False):
    if reverse:
        return np.einsum("bji,bi->bj", gauges, x)
    else:
        return np.einsum("bij,bi->bj", gauges, x)


def local_to_global(x, gauges):
    return np.einsum("bj,bij->bi", x, gauges)


def node_eigencoords(node_ind, evecs_Lc, dim):
    r, c = evecs_Lc.shape
    evecs_Lc = evecs_Lc.reshape(-1, c*dim)
    node_coords = evecs_Lc[node_ind]
    return node_coords.reshape(-1, c)


def posterior(kernel, X_s, X_train, Y_train, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
        
    K = kernel.K(X_train, X_train).numpy()
    K += sigma_y**2 * np.eye(len(K))
    K_s = kernel.K(X_train, X_s).numpy()
    K_ss = kernel.K(X_s, X_s).numpy() 
    K_ss += 1e-8 * np.eye(len(K_ss))
    K_inv = inv(K)
    
    r, c = Y_train.shape
    Y_train = Y_train.reshape(-1,1)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    mu_s = mu_s.reshape(-1, c)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s