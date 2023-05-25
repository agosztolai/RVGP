#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


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