import math


import numpy as np
import jax.numpy as jnp

from einops import rearrange

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.tri as tri
import matplotlib

matplotlib.rcParams["text.usetex"] = True


def plot_scalar_field(
    X,
    Y,
    ax=None,
    colormap="viridis",
    zorder=1,
    n_axis=50,
    levels=8,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    X_1, X_2 = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), num=n_axis),
        np.linspace(X[:, 1].min(), X[:, 1].max(), num=n_axis),
    )

    triang = tri.Triangulation(X[:, 0], X[:, 1])
    interpolator = tri.LinearTriInterpolator(triang, Y)
    Z = interpolator(X_1, X_2)

    ax.contourf(X_1, X_2, Z, cmap=colormap, zorder=zorder, levels=levels)

    return ax


def plot_vector_field(
    X, Y, ax=None, color=None, scale=15, width=None, label=None, zorder=1
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    if color is None:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            np.hypot(Y[:, 0], Y[:, 1]),
            scale=scale,
            width=width,
            label=label,
            zorder=zorder,
            pivot="mid",
        )
    else:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            color=color,
            scale=scale,
            width=width,
            label=label,
            zorder=zorder,
            pivot="mid",
        )

    return ax


def plot_covariances(
    X,
    covariances,
    ax=None,
    alpha=0.5,
    color="cyan",
    edgecolor="k",
    scale=0.8,
    label=None,
    zorder=0,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
        x_lim = None
        y_lim = None
    else:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

    for i in range(X.shape[0]):
        A = covariances[i]
        if len(A.shape) == 1:
            A = jnp.diag(A)

        eigen_decomp = jnp.linalg.eig(A)
        u = eigen_decomp[1][:, 0]

        angle = 360 * jnp.arctan(u[1] / u[0]) / (2 * jnp.pi)

        if (eigen_decomp[0] < 0).sum() > 0:
            print("Error: Ill conditioned covariance in plot. Skipping")
            continue

        # Get the width and height of the ellipses (eigenvalues of A):
        D = jnp.sqrt(eigen_decomp[0])

        # Plot the Ellipse:
        E = Ellipse(
            xy=X[
                i,
            ],
            width=jnp.real(scale * D[0]),
            height=jnp.real(scale * D[1]),
            angle=jnp.real(angle),
            color=color,
            linewidth=1,
            alpha=alpha,
            # edgecolor=edgecolor,
            # facecolor="none",
            zorder=zorder,
        )
        ax.add_patch(E)

    if label is not None:
        label_ellipse = Ellipse(
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            label=label,
            xy=0,
            width=1,
            height=1,
        )


def plot_inference(
    X_context,
    Y_context,
    X_prediction,
    mean_prediction=None,
    covariance_prediction=None,
    title="",
    size_scale=2,
    ellipse_scale=0.8,
    quiver_scale=60,
    ax=None,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    plot_vector_field(
        X_context, Y_context, ax=ax, color="red", scale=quiver_scale, zorder=2
    )
    if mean_prediction is not None:
        plot_vector_field(
            X_prediction, mean_prediction, ax=ax, scale=quiver_scale, zorder=1
        )
    if covariance_prediction is not None:
        plot_covariances(
            X_prediction, covariance_prediction, ax, scale=ellipse_scale, zorder=0
        )


def plot_mean_cov(
    X,
    Y_mean,
    Y_cov,
    title="",
    size_scale=2,
    ellipse_scale=0.8,
    quiver_scale=60,
    ax=None,
):

    if ax == None:
        fig, ax = plt.subplots(1, 1)
    plot_vector_field(X, Y_mean, ax=ax, scale=quiver_scale, zorder=1)
    plot_covariances(X, Y_cov, ax=ax, scale=ellipse_scale, zorder=0)


def plot_2d_sparse_gp(
    sparse_gp,
    sparse_gp_params,
    sparse_gp_state,
    m_cond=None,
    v_cond=None,
    ground_truth_locs=None,
    ground_truth_vals=None,
    scale=200,
    ax=None,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    if (m_cond is not None) and (v_cond is not None):
        plot_vector_field(m_cond, v_cond, ax=ax, scale=scale, color="red", zorder=1)

    inducing_means = sparse_gp.get_inducing_mean(sparse_gp_params, sparse_gp_state)
    plot_vector_field(
        sparse_gp_params.inducing_locations,
        inducing_means,
        color="green",
        scale=scale,
        zorder=4,
        ax=ax,
    )

    if (ground_truth_locs is not None) and (ground_truth_vals is not None):
        plot_vector_field(
            ground_truth_locs,
            ground_truth_vals,
            color="black",
            scale=scale,
            zorder=2,
            ax=ax,
        )

    if ground_truth_locs is not None:

        posterior_samples = sparse_gp(
            sparse_gp_params, sparse_gp_state, ground_truth_locs
        )

        for i in range(posterior_samples.shape[0]):
            plot_vector_field(
                ground_truth_locs,
                posterior_samples[i],
                color="grey",
                scale=scale,
                zorder=1,
                ax=ax,
            )
