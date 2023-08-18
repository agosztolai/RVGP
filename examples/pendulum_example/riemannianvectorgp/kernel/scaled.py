from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

# import tensorflow_probability

# from tensorflow_probability.python.internal.backend import jax as tf2jax

# tfp = tensorflow_probability.experimental.substrates.jax
# tfk = tfp.math.psd_kernels

from .kernel import AbstractKernel


class ScaledKernelParams(NamedTuple):
    log_amplitude: jnp.ndarray
    sub_kernel_params: NamedTuple


class ScaledKernel(AbstractKernel):
    def __init__(
        self,
        sub_kernel: AbstractKernel,
    ):
        self.sub_kernel = sub_kernel
        self.input_dimension = sub_kernel.input_dimension
        self.output_dimension = sub_kernel.output_dimension

    def init_params(
        self,
        key: jnp.ndarray,
    ) -> NamedTuple:
        log_amplitudes = jnp.zeros((1))
        sub_kernel_params = self.sub_kernel.init_params(key)

        return ScaledKernelParams(log_amplitudes, sub_kernel_params)

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: NamedTuple,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        sub_k = self.sub_kernel.matrix(params.sub_kernel_params, x1, x2)
        amplitude = jnp.exp(params.log_amplitude)
        return amplitude * sub_k

    @partial(jit, static_argnums=(0, 3))
    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        return self.sub_kernel.sample_fourier_features(
            params.sub_kernel_params, key, num_samples
        )

    @partial(jit, static_argnums=(0,))
    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:
        return self.sub_kernel.weight_variance(params.sub_kernel_params, state)

    @partial(jit, static_argnums=(0,))
    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        root_amplitude = jnp.exp(0.5 * params.log_amplitude)
        sub_kernel_basis_functions = self.sub_kernel.basis_functions(
            params.sub_kernel_params, state, x
        )
        return root_amplitude * sub_kernel_basis_functions
