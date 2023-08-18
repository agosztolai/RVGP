from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

# import tensorflow_probability

# from tensorflow_probability.python.internal.backend import jax as tf2jax

from riemannianvectorgp.manifold import AbstractEmbeddedRiemannianManifold
from .kernel import AbstractKLKernel, AbstractKernel, EigenBasisFunctionState
from .utils import pairwise_dimension_distance

# tfp = tensorflow_probability.experimental.substrates.jax
# tfk = tfp.math.psd_kernels


class ManifoldProjectionVectorKernel(AbstractKLKernel):
    def __init__(
        self,
        euclidean_vector_kernel: AbstractKernel,
        manifold: AbstractEmbeddedRiemannianManifold,
    ):
        assert (
            manifold.embedded_dimension == euclidean_vector_kernel.output_dimension
        ) or (
            euclidean_vector_kernel.output_dimension == 1
        ), f"Kernel {euclidean_vector_kernel} output dim does not match the embedding dim of the manifold {manifold}"

        self.euclidean_vector_kernel = euclidean_vector_kernel
        self.manifold = manifold

        self.input_dimension = self.euclidean_vector_kernel.input_dimension
        self.output_dimension = self.manifold.dimension

    def init_params(
        self,
        key: jnp.ndarray,
    ) -> NamedTuple:
        return self.euclidean_vector_kernel.init_params(key)

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: NamedTuple,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        Ke = self.euclidean_vector_kernel.matrix(params, x1, x2)
        p1 = self.manifold.projection_matrix(x1)
        p2 = self.manifold.projection_matrix(x2)

        if self.euclidean_vector_kernel.output_dimension == 1:
            # print(f"{p1.shape=}")
            # print(f"{p2.shape=}")
            # print(f"{jnp.eye(self.manifold.embedded_dimension).shape=}")
            tangent_projection = jnp.einsum(
                "...iem,ef,...jfn->...ijmn",
                p1,
                jnp.eye(self.manifold.embedded_dimension),
                p2,
            )
            # print(f"{tangent_projection.shape=}")
            # print(f"{Ke.shape=}")
            K = Ke * tangent_projection
        else:
            K = jnp.einsum("...iem,...ijef,...jfn->...jinm", p1, Ke, p2)

        return K

    @partial(jit, static_argnums=(0, 3))
    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        return self.euclidean_vector_kernel.sample_fourier_features(
            params, key, num_samples
        )

    @partial(jit, static_argnums=(0,))
    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:
        return jnp.repeat(
            self.euclidean_vector_kernel.weight_variance(params, state),
            self.manifold.embedded_dimension,
            axis=-1,
        )

    @partial(jit, static_argnums=(0,))
    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        basis_functions = self.euclidean_vector_kernel.basis_functions(params, state, x)
        p = self.manifold.projection_matrix(x)
        # print(basis_functions.shape)
        # print(p.shape)
        return jnp.einsum("...nbem,...nem->...nbme", basis_functions, p)
