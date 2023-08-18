from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import jax.numpy as jnp

# from tensorflow_probability.python.internal.backend import jax as tf2jax
import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

from .manifold import AbstractRiemannianMainfold, ProductManifold


class AbstractEmbeddedRiemannianManifold(AbstractRiemannianMainfold):
    """Abstract class for a manifold embedded in $\mathbb{R}^n$"""

    embedded_dimension = None

    @abstractmethod
    def m_to_e(self, M):
        """Maps points from intrinsic coordinates to Euclidean coordinates

        Parameters
        ----------
        M : jnp.array
            array of intrinsic coordinates
        """
        pass

    @abstractmethod
    def e_to_m(self, E):
        """Maps points from Euclidean coordinates to intrinsic coordinates

        Parameters
        ----------
        E : jnp.array
        """
        pass

    @partial(jit, static_argnums=(0,))
    def projection_matrix(self, M):
        """Matrix function that projects intrinsic tangent vectors into
        the ambient Euclidean space. Auto computed as the Jacobian of the
        embedding into Euclidean space.

        Implicitly encodes a choice of gauge.

        Parameters
        ----------
        M : jnp.array
        """
        grad_proj = jnp.stack(
            [
                jax.vmap(jax.grad(lambda m: self.m_to_e(m)[..., i]))(M)
                for i in range(self.embedded_dimension)
            ],
            axis=-2,
        )

        return grad_proj / jnp.linalg.norm(grad_proj, axis=-2)[..., np.newaxis, :]

    def mesh(self, n_points):
        """Generates an intrinsic coordinate mesh of the manifold with some
        discretisation level.

        Parameters
        ----------
        n_points : int.
            discretisation level. Usually (n_points + 1)^d vertices
        """
        raise NotImplementedError()

    def projection_matrix_to_3d(self, M):
        """Projection matrix to project tangent space vectors to 3d vectors
        for visualisation

        Parameters
        ----------
        M : jnp.ndarray
        """
        raise NotImplementedError()

    def m_to_3d(self, M):
        """Projects intrinsic coordinates to 3D coordinates for visualisation

        Parameters
        ----------
        M : jnp.ndarray
        """
        raise NotImplementedError()

    @partial(jit, static_argnums=(0,))
    def project_to_m(self, X, Y):
        """Projects a set of Euclidean locations and vectors to the intrinsic
        coordinates and tangent spaces.

        Parameters
        ----------
        X : jnp.ndarray
            Euclidean coordinates
        Y : jnp.ndarray
            Euclidean vectors

        Returns
        -------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tangent vectors in gauge defined by self.projection_matrix
        """
        M = self.e_to_m(X)
        return (
            M,
            (jnp.swapaxes(self.projection_matrix(M), -1, -2) @ Y[..., np.newaxis])[
                ..., 0
            ],
        )

    @partial(jit, static_argnums=(0,))
    def project_to_e(self, M, V):
        """Projects a set of intrinsic coordinates and tangent vectors to
        euclidean space.

        Parameters
        ----------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tangent vectors

        Returns
        -------
        X : jnp.ndarray
            Euclidean coordinates
        Y : jnp.ndarray
            Euclidean vectors
        """
        return (
            self.m_to_e(M),
            (self.projection_matrix(M) @ V[..., np.newaxis])[..., 0],
        )

    @partial(jit, static_argnums=(0,))
    def tanget_projection(self, M1, M2):
        p1 = self.projection_matrix(M1)
        p2 = self.projection_matrix(M2)
        # return jnp.einsum("...iem,...jfn->...jinm", p1, p2)
        return (
            jnp.swapaxes(p2[..., :, np.newaxis, :, :], -1, -2)
            @ p1[..., np.newaxis, :, :, :],
        )

    @partial(jit, static_argnums=(0,))
    def euclidean_projection_matrix(self, M):
        proj_mat = self.projection_matrix(M)
        return jnp.einsum("...em,...fn->...ef", proj_mat, proj_mat)

    @partial(jit, static_argnums=(0,))
    def flatten_to_manifold(self, X, Y):
        pi = self.euclidean_projection_matrix(self.e_to_m(X))
        return jnp.einsum("...ij,...j->...i", pi, Y)

    @partial(jit, static_argnums=(0,))
    def rotate_points(self, M, V, R):
        """Rotate a set of intrinsic coordinates and tangent vectors by a rotation R

        Parameters
        ----------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarry
            Tangent vectors
        R : jnp.ndarray
            Rotation matrix

        Returns
        -------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tangent vectors in gauge defined by self.projection_matrix
        """
        X = self.m_to_e(M)
        X_ = (R @ X[..., np.newaxis]).squeeze(-1)
        M_ = self.e_to_m(X_)
        vector_transport_map = (
            self.projection_matrix(M_)
            @ R
            @ jnp.swapaxes(self.projection_matrix(M), -1, -2)
        )
        V_ = (vector_transport_map @ V[..., np.newaxis]).squeeze(-1)
        return M_, V_

    @partial(jit, static_argnums=(0,))
    def project_to_3d(self, M, V):
        """Projects intrinsic coordinates and tangent vectors to 3D for visualisation

        Parameters
        ----------
        M : jnp.ndarray
            Intrinsic coordinates
        V : jnp.ndarray
            Tanget vectors

        Returns
        -------
        X : jnp.ndarray
            3D coordinates
        Y : jnp.ndarray
            3D vectors
        """
        X = self.m_to_3d(M)
        Y = (
            jnp.swapaxes(self.projection_matrix_to_3d(M), -1, -2) @ V[..., np.newaxis]
        ).squeeze()
        return X, Y

    def __mul__(self, other):
        if isinstance(other, AbstractEmbeddedRiemannianManifold):
            return EmbeddedProductManifold(self, other)
        elif isinstance(other, EmbeddedProductManifold):
            return EmbeddedProductManifold(
                self, *other.sub_manifolds, num_eigenfunctions=other.num_eigenfunctions
            )
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        if isinstance(other, AbstractEmbeddedRiemannianManifold):
            return EmbeddedProductManifold(other, self)
        elif isinstance(other, EmbeddedProductManifold):
            return EmbeddedProductManifold(
                *other.sub_manifolds, self, num_eigenfunctions=other.num_eigenfunctions
            )
        else:
            raise NotImplementedError()


class EmbeddedProductManifold(AbstractEmbeddedRiemannianManifold, ProductManifold):
    def __init__(self, *sub_manifolds, **kwargs):
        super().__init__(*sub_manifolds, **kwargs)
        self.sub_embedded_dimensions = [
            sm.embedded_dimension for sm in self.sub_manifolds
        ]
        self.embedded_dimension = sum(self.sub_embedded_dimensions)

    @partial(jit, static_argnums=(0,))
    def m_to_e(self, M):
        sub_M = jnp.split(M, self.sub_dimensions[:-1], axis=-1)
        sub_X = [
            self.sub_manifolds[i].m_to_e(sub_M[i])
            for i in range(len(self.sub_manifolds))
        ]
        return jnp.concatenate(sub_X, axis=-1)

    @partial(jit, static_argnums=(0,))
    def e_to_m(self, E):
        sub_E = jnp.split(E, self.sub_dimensions[:-1], axis=-1)
        sub_M = [
            self.sub_manifolds[i].e_to_m(sub_E[i])
            for i in range(len(self.sub_manifolds))
        ]
        return jnp.concatenate(sub_M, axis=-1)

    @partial(jit, static_argnums=(0,))
    def projection_matrix(self, M):
        M_shape = M.shape
        M = M.reshape((-1, M_shape[-1]))
        sub_M = jnp.split(M, self.sub_dimensions[:-1], axis=-1)
        sub_projection_matricies = [
            self.sub_manifolds[i].projection_matrix(sub_M[i])
            for i in range(len(self.sub_manifolds))
        ]
        block_diag = jax.vmap(jsp.linalg.block_diag)(*sub_projection_matricies)
        return block_diag.reshape((*M_shape[:-1], *block_diag.shape[-2:]))

    def __mul__(self, other):
        if isinstance(other, AbstractEmbeddedRiemannianManifold):
            return EmbeddedProductManifold(*self.sub_manifolds, other)
        elif isinstance(other, EmbeddedProductManifold):
            return EmbeddedProductManifold(
                *self.sub_manifolds,
                *other.sub_manifolds,
                num_eigenfunctions=max(
                    self.num_eigenfunctions, other.num_eigenfunctions
                )
            )
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        if isinstance(other, AbstractEmbeddedRiemannianManifold):
            return EmbeddedProductManifold(other, *self.sub_manifolds)
        elif isinstance(other, EmbeddedProductManifold):
            return EmbeddedProductManifold(
                *other.sub_manifolds,
                *self.sub_manifolds,
                num_eigenfunctions=max(
                    self.num_eigenfunctions, other.num_eigenfunctions
                )
            )
        else:
            raise NotImplementedError()
