from functools import partial

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

from riemannianvectorgp.utils.spherical_harmonics import (
    _spherical_harmonics,
    _d_n,
    _c_nd,
    _c_n,
)
from riemannianvectorgp.utils import projection_matrix
from .manifold import AbstractRiemannianMainfold
from .embedded_manifold import AbstractEmbeddedRiemannianManifold


class S2(AbstractRiemannianMainfold):
    dimension = 2
    compact = True

    def __init__(
        self,
        radius: float = 1.0,
        max_l: int = 11,
        # spectrum_func = None,
    ):
        self.radius = radius
        self.max_l = max_l
        self.constants = np.zeros(
            sum([_d_n(n, self.dimension) for n in range(max_l + 1)])
        )

        i = 0
        for n in range(self.max_l + 1):
            d_n = _d_n(n, self.dimension)
            cn = _c_n(n)
            self.constants[i : (i + d_n)] = jnp.sqrt(cn)
            i += d_n

        self.constants = jnp.array(self.constants)

    # def compute_normalised_constants(self, spectrum_func):
    #     self.constants = np.zeros(
    #         sum([_d_n(n, self.dimension) for n in range(self.max_l + 1)])
    #     )

    #     eig_vals = []
    #     cns = []

    #     i = 0
    #     for n in range(self.max_l + 1):
    #         d_n = _d_n(n, self.dimension)
    #         cn = _c_n(n)
    #         self.constants[i : (i + d_n)] = jnp.sqrt(cn)
    #         cns.append(cn)
    #         eig_vals.append(self.laplacian_eigenvalue(i))
    #         i += d_n

    #     eig_vals = jnp.array(eig_vals)

    #     psds = spectrum_func(eig_vals)
    #     c_nu = jnp.sum(psds * jnp.array(cns))
    #     self.constants = self.constants # / c_nu

    #     print(c_nu)

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenvalue(
        self,
        n: int,
    ) -> jnp.ndarray:
        l = jnp.floor(jnp.sqrt(n))
        return l * (l + 1)

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenfunction(
        self,
        n: int,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        phi = x[..., 0]
        theta = x[..., 1]

        x = jnp.cos(theta) * jnp.sin(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(phi)

        e = jnp.stack([x, y, z], axis=-1)

        return (
            self.constants[n, np.newaxis]
            * _spherical_harmonics(self.max_l, e)[..., n, np.newaxis]
        )

    def __repr__(
        self,
    ):
        if self.radius == 1.0:
            return f"\U0001D4E2\u2082"
        else:
            return f"\U0001D4E2\u2082({self.radius})"


class EmbeddedS2(AbstractEmbeddedRiemannianManifold, S2):
    embedded_dimension = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def m_to_e(self, M):
        phi = M[..., 0]
        theta = M[..., 1]

        return jnp.stack(
            [
                jnp.sin(phi) * jnp.cos(theta),
                jnp.sin(phi) * jnp.sin(theta),
                jnp.cos(phi),
            ],
            axis=-1,
        )

    @partial(jit, static_argnums=(0,))
    def e_to_m(self, E):
        x = E[..., 0]
        y = E[..., 1]
        z = E[..., 2]
        return jnp.stack([jnp.arccos(z), jnp.arctan2(y, x)], axis=-1)

    @partial(jit, static_argnums=(0,))
    def projection_matrix(self, M):
        # phi = M[..., 0]
        # theta = M[..., 1]

        # e1 = jnp.stack(
        #     [
        #         jnp.cos(phi) * jnp.cos(theta),
        #         jnp.cos(phi) * jnp.sin(theta),
        #         -jnp.sin(phi),
        #     ],
        #     axis=-1,
        # )
        # e2 = -jnp.stack([-jnp.sin(theta), jnp.cos(theta), jnp.zeros_like(phi)], axis=-1)

        # return jnp.stack(
        #     [
        #         e1,
        #         e2,
        #     ],
        #     axis=-1,
        # )
        return projection_matrix(M, self.m_to_e)
