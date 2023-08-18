from functools import partial
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

from riemannianvectorgp.utils import _spherical_harmonics
from riemannianvectorgp.manifold import (
    AbstractRiemannianMainfold,
    S1,
    AbstractEmbeddedRiemannianManifold,
    EmbeddedS1,
)


class AbstractDoubleCoverRiemannianManifold(AbstractRiemannianMainfold):
    double_cover = None
    test_points = None

    def __init__(self, precompute_eigenfunctions=1000, tol=1e-3):
        self.eigenindicies = self.precompute_eigenfunctions(
            precompute_eigenfunctions, tol
        )

    def precompute_eigenfunctions(self, num_eigenfunctions, tol):
        eigenindicies = jnp.arange(num_eigenfunctions)
        diff = self.double_cover.laplacian_eigenfunction(
            eigenindicies, self.test_points
        ) - self.double_cover.laplacian_eigenfunction(
            eigenindicies, self.identification(self.test_points)
        )
        test = jnp.mean(jnp.abs(diff), axis=(0, 2))

        indices = test < tol
        return eigenindicies[indices]

    def laplacian_eigenvalue(self, n):
        return self.double_cover.laplacian_eigenvalue(self.eigenindicies[n])

    def laplacian_eigenfunction(self, n, x):
        return self.double_cover.laplacian_eigenfunction(self.eigenindicies[n], x)

    @abstractmethod
    def identification(self, M: jnp.array):
        pass


class AbstractEmbeddedDoubleCoverRiemannianManifold(
    AbstractDoubleCoverRiemannianManifold, AbstractEmbeddedRiemannianManifold
):
    pass


class KleinBottle(AbstractDoubleCoverRiemannianManifold):
    dimension = 2

    def __init__(self, r1=0.5, r2=None, **kwargs):
        if r2 is None:
            r2 = 2 * r1

        self.double_cover = S1(r1) * S1(r2)

        # TODO: Better code?
        num_points = 30
        u = np.linspace(0, np.pi, num_points + 1)[1:]
        v = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
        u, v = np.meshgrid(u, v, indexing="ij")
        u = u.flatten()
        v = v.flatten()
        self.test_points = np.stack([u, v], axis=-1)

        super().__init__(**kwargs)

    def identification(self, M):
        u, v = M[..., 0], M[..., 1]
        return jnp.stack([u + jnp.pi, 2 * jnp.pi - v], axis=-1)

    def __repr__(
        self,
    ):
        if (self.double_cover.sub_manifolds[0].radius == 0.5) and (
            self.double_cover.sub_manifolds[1].radius == 1.0
        ):
            # return f"\U0001D4DA\U0001D4D1"
            return f"\U0001D4DA\U0001D4F5\U0001D4EE\U0001D4F2\U0001D4F7\U0001D4D1\U0001D4F8\U0001D4FD\U0001D4FD\U0001D4F5\U0001D4EE"
        else:
            return f"\U0001D4DA\U0001D4F5\U0001D4EE\U0001D4F2\U0001D4F7\U0001D4D1\U0001D4F8\U0001D4FD\U0001D4FD\U0001D4F5\U0001D4EE({self.double_cover.sub_manifolds[0].radius},{self.double_cover.sub_manifolds[1].radius})"


class EmbeddedKleinBottle(AbstractEmbeddedDoubleCoverRiemannianManifold, KleinBottle):
    embedded_dimension = 4

    def __init__(self, *args, eps=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps

    @partial(jit, static_argnums=(0,))
    def m_to_e(self, M):
        u, v = M[..., 0], M[..., 1]
        u = u * 2

        cu = jnp.cos(u)
        su = jnp.sin(u)
        cu2 = jnp.cos(u / 2)
        su2 = jnp.sin(u / 2)
        cv = jnp.cos(v)
        sv = jnp.sin(v)
        s2v = jnp.sin(2 * v)

        return jnp.stack(
            [
                cu2 * cv - su2 * s2v,
                su2 * cv + cu2 * s2v,
                cu * (1 + self.eps * sv),
                su * (1 + self.eps * sv),
            ],
            axis=-1,
        )

    @partial(jit, static_argnums=(0,))
    def e_to_m(self, E):
        pass
