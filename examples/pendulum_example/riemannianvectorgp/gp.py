from typing import NamedTuple, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from functools import partial

# import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax

# tfp = tensorflow_probability.experimental.substrates.jax
# tfk = tfp.math.psd_kernels
from abc import ABC, abstractmethod
from .kernel import AbstractKernel, FourierFeatures

from einops import rearrange


class GaussianProcessParameters(NamedTuple):
    kernel_params: NamedTuple


class GaussianProcessState(NamedTuple):
    locations: jnp.ndarray
    values: jnp.ndarray
    noises: jnp.ndarray
    K_inv: jnp.ndarray


class GaussianProcess:
    """A Gaussian process, implemented as a Haiku module."""

    def __init__(
        self,
        kernel: AbstractKernel,
    ):
        """Initializes the GP.

        Args:
            kernel: the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
            num_samples: the number of samples stored in the GP.
        """
        self.kernel = kernel
        self.input_dimension = kernel.input_dimension
        self.output_dimension = kernel.output_dimension

    def init_params_with_state(
        self,
        key: jnp.ndarray,
    ) -> Tuple[GaussianProcessParameters, GaussianProcessState]:
        (k1, k2) = jr.split(key)
        kernel_params = self.kernel.init_params(k1)

        (OD, ID) = (
            self.output_dimension,
            self.input_dimension,
        )
        params = GaussianProcessParameters(
            kernel_params,
        )

        locations = jnp.array([])
        values = jnp.array([])
        noises = jnp.array([])
        K_inv = jnp.array([])
        state = GaussianProcessState(locations, values, noises, K_inv)

        return (params, state)

    @partial(jax.jit, static_argnums=(0,))
    def condition(
        self,
        params: GaussianProcessParameters,
        locations: jnp.ndarray,
        values: jnp.ndarray,
        noises: jnp.ndarray,
    ):
        (OD, N) = (
            self.output_dimension,
            locations.shape[0],
        )
        noises_ = rearrange(noises, "N OD -> (N OD)")
        K = self.kernel.matrix(params.kernel_params, locations, locations)
        K = rearrange(K, "N1 N2 OD1 OD2 -> (N1 OD1) (N2 OD2)")
        K = K + jnp.diag(noises_)
        (cholesky, _) = jsp.linalg.cho_factor(
            K,
            lower=True,
        )
        K_inv = tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(
            tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(
                jnp.eye(N * OD)
            ),
            adjoint=True,
        )
        # K_inv = tf2jax.linalg.inv(K)
        K_inv = rearrange(
            K_inv, "(N1 OD1) (N2 OD2) -> N1 N2 OD1 OD2", N1=N, N2=N, OD1=OD, OD2=OD
        )

        return GaussianProcessState(locations, values, noises, K_inv)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: GaussianProcessParameters,
        state: GaussianProcessState,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (OD, ID, M) = (
            self.output_dimension,
            self.input_dimension,
            x.shape[0],
        )
        K_inv = rearrange(state.K_inv, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")
        values = rearrange(state.values, "M OD -> (M OD)")
        kernel_params = params.kernel_params

        K_sn = self.kernel.matrix(kernel_params, x, state.locations)
        K_ns = self.kernel.matrix(kernel_params, state.locations, x)
        K_ss = self.kernel.matrix(kernel_params, x, x)

        K_sn = rearrange(K_sn, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")
        K_ns = rearrange(K_ns, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")
        K_ss = rearrange(K_ss, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")

        # print(f"{K_inv.shape=}")
        # print(f"{K_sn.shape=}")
        # print(f"{K_ns.shape=}")
        # print(f"{K_ss.shape=}")
        m = tf2jax.linalg.matvec(K_sn, tf2jax.linalg.matvec(K_inv, values))
        K = K_ss - tf2jax.linalg.matmul(tf2jax.linalg.matmul(K_sn, K_inv), K_ns)

        m = rearrange(m, "(M OD) -> M OD", M=M, OD=OD)
        K = rearrange(
            K, "(M1 OD1) (M2 OD2) -> M1 M2 OD1 OD2", M1=M, M2=M, OD1=OD, OD2=OD
        )

        return m, K

    @partial(jax.jit, static_argnums=(0, 4))
    def sample(
        self,
        params: GaussianProcessParameters,
        state: GaussianProcessState,
        x: jnp.ndarray,
        num_samples: int,
        key: jnp.ndarray,
        obs_noise: float = 1e-6,
    ) -> jnp.ndarray:
        m, K = self(params, state, x)
        M, OD = m.shape
        m = rearrange(m, "M OD -> (M OD)")
        K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")
        cholesky = jsp.linalg.cho_factor(
            K + jnp.identity(M * OD) * obs_noise, lower=True
        )[0]
        S = num_samples
        sample_noise = jr.normal(key, (S, M * OD))
        samples = m + tf2jax.linalg.matvec(cholesky, sample_noise)
        samples = rearrange(samples, "S (M OD) -> S M OD", M=M, OD=OD)
        return samples

    # @partial(jax.jit, static_argnums=(0,))
    # def randomize(
    #     self,
    #     params: GaussianProcessParameters,
    #     state: GaussianProcessState,
    #     key: jnp.ndarray,
    # ) -> GaussianProcessState:
    #     """Samples a new set of random functions from the GP."""
    #     (S, N) = (
    #         self.num_samples,
    #         state.locations.shape[0],
    #     )
    #     sample_noise = jr.normal(key, (S, N))
    #     state = state._replace(sample_noise=sample_noise)
    #     return state

    # @partial(jax.jit, static_argnums=(0,))
    # def prior_kl(
    #     self,
    #     params: SparseGaussianProcessParameters,
    #     state: SparseGaussianProcessState,
    # ) -> jnp.ndarray:
    #     """Evaluates the prior KL term in the sparse VI objective."""
    #     (OD, ID, M) = (self.output_dimension, self.input_dimension, self.num_inducing)
    #     inducing_locations = params.inducing_locations
    #     inducing_pseudo_mean = params.inducing_pseudo_mean
    #     inducing_pseudo_log_err_stddev = params.inducing_pseudo_log_err_stddev
    #     kernel_params = params.kernel_params
    #     cholesky = state.cholesky

    #     logdet_term = 2 * jnp.sum(jnp.log(jax.vmap(jnp.diag)(cholesky))) - 2 * jnp.sum(
    #         inducing_pseudo_log_err_stddev
    #     )
    #     kernel_matrix = self.kernel.matrix(
    #         kernel_params, inducing_locations, inducing_locations
    #     )
    #     cholesky_inv = tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(
    #         tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(jnp.eye(M)),
    #         adjoint=True,
    #     )
    #     trace_term = jnp.sum(cholesky_inv * kernel_matrix)
    #     reparameterized_quadratic_form_term = jnp.sum(
    #         inducing_pseudo_mean
    #         * tf2jax.linalg.matvec(kernel_matrix, inducing_pseudo_mean)
    #     )
    #     return (
    #         logdet_term
    #         - (OD * ID * M)
    #         + trace_term
    #         + reparameterized_quadratic_form_term
    #     ) / 2

    # @partial(jax.jit, static_argnums=(0,))
    # def hyperprior(
    #     self,
    #     params: SparseGaussianProcessParameters,
    #     state: SparseGaussianProcessState,
    # ) -> jnp.ndarray:
    #     """Returns the log hyperprior regularization term of the GP."""
    #     return jnp.zeros(())  # temporary

    # @partial(jax.jit, static_argnums=(0,))
    # def loss(
    #     self,
    #     params: SparseGaussianProcessParameters,
    #     state: SparseGaussianProcessState,
    #     key: jnp.ndarray,
    #     x: jnp.ndarray,
    #     y: jnp.ndarray,
    #     n_data: int,
    # ) -> Tuple[jnp.ndarray, SparseGaussianProcessState]:
    #     state = self.randomize(params, state, key)

    #     kl = self.prior_kl(params, state)

    #     f = self(params, state, x)
    #     s = jnp.exp(params.log_error_stddev)
    #     (n_samples, _, n_batch) = f.shape
    #     c = n_data / (n_batch * n_samples * 2)
    #     l = n_data * jnp.sum(jnp.log(s)) + c * jnp.sum(((y - f) / s) ** 2)

    #     r = self.hyperprior(params, state)
    #     return (kl + l + r, state)
