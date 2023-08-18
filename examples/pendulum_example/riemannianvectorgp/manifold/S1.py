from functools import partial

import numpy as np
from jax import jit
import jax.numpy as jnp

from .manifold import AbstractRiemannianMainfold
from .embedded_manifold import AbstractEmbeddedRiemannianManifold


class S1(AbstractRiemannianMainfold):
    dimension = 1
    compact = True

    def __init__(
        self,
        radius: float = 0.5,
    ):
        self.radius = radius

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenvalue(
        self,
        n: int,
    ) -> jnp.ndarray:
        freq = n // 2
        return jnp.power(2 * jnp.pi * freq * self.radius, 2)

    @partial(jit, static_argnums=(0,))
    def laplacian_eigenfunction(
        self,
        n: int,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        freq = n // 2
        phase = -(jnp.pi / 2) * (n % 2)
        return jnp.sqrt(2) * jnp.expand_dims(jnp.cos(x * freq + phase), -1)

    def __repr__(
        self,
    ):
        if self.radius == 1.0:
            return f"\U0001D4E2\u2081"
        else:
            return f"\U0001D4E2\u2081({self.radius})"


class EmbeddedS1(AbstractEmbeddedRiemannianManifold, S1):
    embedded_dimension = 2

    def __init__(
        self,
        radius: float = 1.0,
    ):
        super().__init__(radius)

    @partial(jit, static_argnums=(0,))
    def m_to_e(self, M):
        return self.radius * jnp.concatenate([jnp.sin(M), jnp.cos(M)], axis=-1)

    @partial(jit, static_argnums=(0,))
    def e_to_m(self, E):
        return (
            jnp.arctan2(E[..., 0] / self.radius, E[..., 1] / self.radius) % (2 * jnp.pi)
        )[..., np.newaxis]

    # @partial(jit, static_argnums=(0,))
    # def projection_matrix(self, M):
    #     return jnp.stack([jnp.cos(M), -jnp.sin(M)], axis=-2)
