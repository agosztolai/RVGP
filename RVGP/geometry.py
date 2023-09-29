#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import networkx as nx
import tensorflow as tf
from scipy import sparse

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KDTree



def compute_laplacian(G, normalization=False):

    if normalization:
        laplacian = sparse.csr_matrix(nx.normalized_laplacian_matrix(G), dtype=np.float64)
    else:
        laplacian = sparse.csr_matrix(nx.laplacian_matrix(G), dtype=np.float64)
    
    return laplacian


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


def compute_spectrum(laplacian, n_eigenpairs=None, dtype=tf.float64):
    
    if n_eigenpairs is None:
        n_eigenpairs = laplacian.shape[0]
    if n_eigenpairs >= laplacian.shape[0]:
        n_eigenpairs = laplacian.shape[0]

    evals, evecs = scipy.sparse.linalg.eigsh(laplacian, k=n_eigenpairs, which="SM")
    
    evecs *= np.sqrt(len(evecs))

    evals = tf.convert_to_tensor(evals, dtype=dtype)
    evecs = tf.convert_to_tensor(evecs, dtype=dtype)
    
    return evals, evecs


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


def manifold_dimension(Sigma, frac_explained=0.9):
    """Estimate manifold dimension based on singular vectors"""

    if frac_explained == 1.0:
        return Sigma.shape[1]

    Sigma **= 2
    Sigma /= Sigma.sum(1, keepdims=True)
    Sigma = Sigma.cumsum(1)
    var_exp = Sigma.mean(0) - Sigma.std(0)
    dim_man = np.where(var_exp >= frac_explained)[0][0] + 1

    print("\nFraction of variance explained: ", var_exp)

    return int(dim_man)


def manifold_graph(X, typ = 'knn', n_neighbors=5):
    """Fit graph over a pointset X"""
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

    return G


def find_nn(x_query, X, nn=3, r=None):
    """
    Find nearest neighbors of a point on the manifold

    Parameters
    ----------
    ind_query : 2d np array, list[2d np array]
        Index of points whose neighbors are needed.
    x : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space.
    nn : int, optional
        Number of nearest neighbors. The default is 1.
        
    Returns
    -------
    dist : list[list]
        Distance of nearest neighbors.
    ind : list[list]
        Index of nearest neighbors.

    """
    
    #Fit neighbor estimator object
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    
    if r is not None:
        ind, dist = kdt.query_radius(x_query, r=r, return_distance=True, sort_results=True)
        ind = ind[0]
        dist = dist[0]
    else:
        # apparently, the outputs are reversed here compared to query_radius()
        dist, ind = kdt.query(x_query, k=nn)
            
    return dist, ind.flatten()


def closest_manifold_point(x_query, d, nn=3):
    dist, ind = find_nn(x_query, d.vertices, nn=nn)
    w = 1/(dist.T+0.00001)
    w /= w.sum()
    positional_encoding = d.evecs_Lc.reshape(d.n, -1)
    pe_manifold = (positional_encoding[ind]*w).sum(0, keepdims=True)
    x_manifold = d.vertices[ind]
    
    return x_manifold, pe_manifold


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