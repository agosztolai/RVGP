from abc import ABC, ABCMeta, abstractmethod
from functools import reduce
import operator
import jax.numpy as jnp

from jax import jit
from functools import partial


class AbstractRiemannianMainfold(ABC):
    dimension = None
    compact = False

    @abstractmethod
    def laplacian_eigenvalue(
        self,
        n: jnp.ndarray,
    ):
        pass

    @abstractmethod
    def laplacian_eigenfunction(
        self,
        n: jnp.ndarray,
        x: jnp.ndarray,
    ):
        pass

    def __repr__(
        self,
    ):
        return self.__class__.__name__

    def __mul__(self, other):
        if isinstance(other, AbstractRiemannianMainfold):
            return ProductManifold(self, other)
        elif isinstance(other, ProductManifold):
            return ProductManifold(
                self, *other.sub_manifolds, num_eigenfunctions=other.num_eigenfunctions
            )
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        if isinstance(other, AbstractRiemannianMainfold):
            return ProductManifold(other, self)
        elif isinstance(other, ProductManifold):
            return ProductManifold(
                *other.sub_manifolds, self, num_eigenfunctions=other.num_eigenfunctions
            )
        else:
            raise NotImplementedError()


class ProductManifold(AbstractRiemannianMainfold):
    def __init__(self, *sub_manifolds, num_eigenfunctions=1000):
        self.sub_manifolds = sub_manifolds
        self.sub_dimensions = [sm.dimension for sm in self.sub_manifolds]
        self.dimension = sum(self.sub_dimensions)
        self.num_eigenfunctions = num_eigenfunctions

        self.compact = reduce(operator.and_, [m.compact for m in self.sub_manifolds])

        if self.compact:
            # precompute the eigenindicies of the product manifold eigenvalues
            sub_manifold_eigenindicies = [
                jnp.arange(self.num_eigenfunctions) for m in self.sub_manifolds
            ]
            sub_manifold_eigenvalues = [
                m.laplacian_eigenvalue(i)
                for m, i in zip(self.sub_manifolds, sub_manifold_eigenindicies)
            ]
            sub_manifold_eigenindicies = jnp.stack(
                jnp.meshgrid(*sub_manifold_eigenindicies), axis=-1
            ).reshape((-1, len(self.sub_manifolds)))
            sub_manifold_eigenvalues = jnp.sum(
                jnp.stack(jnp.meshgrid(*sub_manifold_eigenvalues), axis=-1), axis=-1
            ).reshape(-1)
            index = jnp.argsort(sub_manifold_eigenvalues)
            self.sub_manifold_eigenindicies = sub_manifold_eigenindicies[index][
                : self.num_eigenfunctions
            ]

    @partial(jit, static_argnums=(0))
    def laplacian_eigenvalue(
        self,
        n: jnp.ndarray,
    ) -> jnp.ndarray:
        # TODO: SHARP EDGE! no check that the eigen index has been precomputed as it breaks jit'ing
        # if jnp.any(n >= self.num_eigenfunctions):
        #     raise ValueError(
        #         f"Max n ({n.max()}) greater than the precomputed eigenvalues ({self.num_eigenfunctions}) for this product manifold"
        #     )
        if self.compact:
            return jnp.sum(
                jnp.stack(
                    [
                        m.laplacian_eigenvalue(self.sub_manifold_eigenindicies[n, i])
                        for i, m in enumerate(self.sub_manifolds)
                    ],
                    axis=-1,
                ),
                axis=-1,
            )
        else:
            raise NotImplementedError()

    @partial(jit, static_argnums=(0))
    def laplacian_eigenfunction(
        self,
        n: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.compact:
            sub_x = jnp.split(x, self.sub_dimensions[:-1], axis=-1)
            return jnp.prod(
                jnp.stack(
                    [
                        m.laplacian_eigenfunction(
                            self.sub_manifold_eigenindicies[n, i], x
                        )
                        for i, (m, x) in enumerate(zip(self.sub_manifolds, sub_x))
                    ],
                    axis=-1,
                ),
                axis=-1,
            )
        else:
            raise NotImplementedError()

    def __repr__(
        self,
    ):
        superscript_map = {
            1: "",
            2: "\U000000B2",
            3: "\U000000B3",
            4: "\U00002074",
            5: "\U00002705",
            6: "\U00002706",
        }

        reprs = []
        counts = []
        for m in self.sub_manifolds:
            m = str(m)
            if len(reprs) == 0:
                reprs.append(m)
                counts.append(1)
            elif reprs[-1] == m:
                counts[-1] = counts[-1] + 1
            else:
                reprs.append(m)
                counts.append(1)

        contracted_reprs = []
        for r, c in zip(reprs, counts):
            contracted_reprs.append(r + superscript_map[c])

        return "×".join(contracted_reprs)

    def __mul__(self, other):
        if isinstance(other, AbstractRiemannianMainfold):
            return ProductManifold(*self.sub_manifolds, other)
        elif isinstance(other, ProductManifold):
            return ProductManifold(
                *self.sub_manifolds,
                *other.sub_manifolds,
                num_eigenfunctions=max(
                    self.num_eigenfunctions, other.num_eigenfunctions
                ),
            )
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        if isinstance(other, AbstractRiemannianMainfold):
            return ProductManifold(other, *self.sub_manifolds)
        elif isinstance(other, ProductManifold):
            return ProductManifold(
                *other.sub_manifolds,
                *self.sub_manifolds,
                num_eigenfunctions=max(
                    self.num_eigenfunctions, other.num_eigenfunctions
                ),
            )
        else:
            raise NotImplementedError()
