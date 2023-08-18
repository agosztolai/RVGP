from functools import partial

import numpy as np
from jax import jit
import jax.numpy as jnp

from .manifold import AbstractRiemannianMainfold
from .embedded_manifold import AbstractEmbeddedRiemannianManifold


class R(AbstractRiemannianMainfold):
    compact = False

    def __init__(self, n: int):
        self.n = n
        self.dimension = n

    def laplacian_eigenvalue(
        self,
        n: jnp.ndarray,
    ):
        raise NotImplementedError(
            "Can't get the Laplacian eignevalues for Euclidean space"
        )

    def laplacian_eigenfunction(
        self,
        n: jnp.ndarray,
        x: jnp.ndarray,
    ):
        raise NotImplementedError(
            "Can't get the Laplacian eignefunctions for Euclidean space"
        )

    def __repr__(self):
        if self.n == 1:
            return "\u211D\u00B9"
        elif self.n == 2:
            return "\u211D\u00B2"
        elif self.n == 3:
            return "\u211D\u00B3"
        elif self.n == 4:
            return "\u211D\u2074"
        elif self.n == 5:
            return "\u211D\u2075"
        elif self.n == 6:
            return "\u211D\u2076"
        elif self.n == 7:
            return "\u211D\u2077"
        elif self.n == 8:
            return "\u211D\u2078"
        elif self.n == 9:
            return "\u211D\u2079"
        else:
            return "\u211D" + str(self.n)


class EmbeddedR(AbstractEmbeddedRiemannianManifold, R):
    def __init__(self, n: int):
        super().__init__(n)
        self.embedded_dimension = self.n

    @partial(jit, static_argnums=(0,))
    def m_to_e(self, M):
        return M

    @partial(jit, static_argnums=(0,))
    def e_to_m(self, E):
        return E

    @partial(jit, static_argnums=(0,))
    def projection_matrix(self, M):
        return jnp.ones_like(M)[..., np.newaxis] * jnp.eye(self.n)
