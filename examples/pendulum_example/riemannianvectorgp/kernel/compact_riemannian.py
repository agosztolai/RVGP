from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

# import tensorflow_probability

from riemannianvectorgp.manifold import AbstractRiemannianMainfold
from .kernel import AbstractKLKernel, EigenBasisFunctionState
from .utils import pairwise_dimension_distance

# tfp = tensorflow_probability.experimental.substrates.jax
# tfk = tfp.math.psd_kernels


class CompactRiemannianManifoldKernel(AbstractKLKernel):
    def __init__(
        self,
        compact_riemannian_manifold: AbstractRiemannianMainfold,
        truncation: int,
    ):
        self.manifold = compact_riemannian_manifold
        self.truncation = truncation
        self.input_dimension = self.manifold.dimension
        self.output_dimension = 1

    @abstractmethod
    def spectrum(
        self,
        eigenvalues: jnp.ndarray,
        params: NamedTuple,
    ) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: NamedTuple,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        eigenindicies = jnp.arange(self.truncation)
        fx1 = self.basis_functions(params, EigenBasisFunctionState(eigenindicies), x1)
        fx2 = self.basis_functions(params, EigenBasisFunctionState(eigenindicies), x2)
        lam = self.manifold.laplacian_eigenvalue(eigenindicies)
        spectrum = self.spectrum(lam, params)

        return jnp.sum(
            fx1[..., np.newaxis, :, :, :]
            * fx2[..., np.newaxis, :, :, :, :]
            * spectrum[:, np.newaxis, np.newaxis],
            axis=-3,
        )

    # static 3 as jnp.arange needs a static length
    @partial(jit, static_argnums=(0, 3))
    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        return EigenBasisFunctionState(jnp.arange(num_samples))

    @partial(jit, static_argnums=(0,))
    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:
        lam = self.manifold.laplacian_eigenvalue(state.eigenindicies)
        return self.spectrum(lam, params)[..., np.newaxis]

    @partial(jit, static_argnums=(0,))
    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.manifold.laplacian_eigenfunction(state.eigenindicies, x)[
            ..., np.newaxis
        ]


class SquaredExponentialCompactRiemannianManifoldKernelParams(NamedTuple):
    log_length_scale: jnp.ndarray


class SquaredExponentialCompactRiemannianManifoldKernel(
    CompactRiemannianManifoldKernel
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_params(
        self,
        key: jnp.ndarray,
    ) -> SquaredExponentialCompactRiemannianManifoldKernelParams:
        log_length_scale = jnp.zeros((1))
        return SquaredExponentialCompactRiemannianManifoldKernelParams(log_length_scale)

    @partial(jit, static_argnums=(0,))
    def spectrum(
        self,
        eigenvalues: jnp.ndarray,
        params: NamedTuple,
    ) -> jnp.ndarray:
        lengthscale2 = jnp.exp(2 * params.log_length_scale)
        return jnp.exp(-(lengthscale2 * eigenvalues / 2))


class MaternCompactRiemannianManifoldKernelParams(NamedTuple):
    log_length_scale: jnp.ndarray


class MaternCompactRiemannianManifoldKernel(CompactRiemannianManifoldKernel):
    def __init__(self, smoothness, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothness = smoothness

    def init_params(
        self,
        key: jnp.ndarray,
    ) -> SquaredExponentialCompactRiemannianManifoldKernelParams:
        log_length_scale = jnp.zeros((1))
        return SquaredExponentialCompactRiemannianManifoldKernelParams(log_length_scale)

    @partial(jit, static_argnums=(0,))
    def spectrum(
        self,
        eigenvalues: jnp.ndarray,
        params: NamedTuple,
    ) -> jnp.ndarray:
        lengthscale = jnp.exp(params.log_length_scale)
        return jnp.power(
            2 * self.smoothness / jnp.power(lengthscale, 2) + eigenvalues,
            -self.smoothness - self.manifold.dimension / 2,
        )
