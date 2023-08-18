from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

# import tensorflow_probability
# from tensorflow_probability.python.internal.backend import jax as tf2jax

# tfp = tensorflow_probability.experimental.substrates.jax
# tfk = tfp.math.psd_kernels


class EigenBasisFunctionState(NamedTuple):
    eigenindicies: jnp.ndarray


class RandomBasisFunctionState(NamedTuple):
    frequency: jnp.ndarray
    phase: jnp.ndarray


class AbstractKernel(ABC):
    @abstractmethod
    def init_params(
        self,
        key: jnp.ndarray,
    ) -> NamedTuple:
        pass

    @abstractmethod
    def matrix(
        self,
        params: NamedTuple,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        pass

    @abstractmethod
    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        pass


class AbstractRFFKernel(AbstractKernel):
    pass


class AbstractKLKernel(AbstractKernel):
    pass


class FourierFeatureState(NamedTuple):
    basis_function_state: NamedTuple
    weights: jnp.ndarray


class FourierFeatures:
    def __init__(
        self,
        kernel: AbstractKernel,
        num_basis_functions: int,
    ):
        self.kernel = kernel
        self.num_basis_functions = num_basis_functions

    def init_state(
        self,
        params: NamedTuple,
        num_samples: int,
        key: jnp.ndarray,
    ) -> NamedTuple:
        basis_function_state = self.kernel.sample_fourier_features(
            params, key, self.num_basis_functions
        )
        weight_variance = self.kernel.weight_variance(params, basis_function_state)
        weight_std = jnp.sqrt(weight_variance)
        weights = weight_std * jr.normal(key, (num_samples, *weight_std.shape))
        return FourierFeatureState(basis_function_state, weights)

    # JAX needs shapes to be static -> num samples needs to be static
    @partial(jit, static_argnums=(0, 3))
    def resample_weights(
        self,
        params: NamedTuple,
        state: NamedTuple,
        num_samples: int,
        key: jnp.ndarray,
    ) -> NamedTuple:
        weight_variance = self.kernel.weight_variance(
            params, state.basis_function_state
        )
        weight_variance = self.kernel.weight_variance(
            params, state.basis_function_state
        )
        weight_std = jnp.sqrt(weight_variance)
        weights = weight_std * jr.normal(key, (num_samples, *weight_variance.shape))
        return FourierFeatureState(state.basis_function_state, weights)

    @partial(jit, static_argnums=(0,))
    def resample_basis(
        self,
        params: NamedTuple,
        state: NamedTuple,
        key: jnp.ndarray,
    ) -> NamedTuple:
        basis_function_state = self.kernel.sample_fourier_features(
            params, key, self.num_basis_functions
        )
        return FourierFeatureState(basis_function_state, state.weights)

    @partial(jit, static_argnums=(0,))
    def __call__(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        basis_functions = self.kernel.basis_functions(
            params,
            state.basis_function_state,
            x,
        )
        weights = state.weights
        # print(f"{weights.shape=}")
        # print(f"{basis_functions.shape=}")
        # return jnp.einsum(
        #     "mloe,sle->smo", basis_functions, weights
        # )  # tf2jax.linalg.matvec(basis_functions, weights)
        if len(x.shape) == 2:
            return jnp.einsum(
                "mloe,sle->smo", basis_functions, weights
            )  # tf2jax.linalg.matvec(basis_functions, weights)
        elif len(x.shape) == 3:
            return jnp.einsum(
                "smloe,sle->smo", basis_functions, weights
            )  # tf2jax.linalg.matvec(basis_functions, weights)
