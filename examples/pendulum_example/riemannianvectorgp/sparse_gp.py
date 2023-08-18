from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from functools import partial

# import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax
from einops import rearrange

# tfp = tensorflow_probability.experimental.substrates.jax
# tfk = tfp.math.psd_kernels
from abc import ABC, abstractmethod
from .kernel import AbstractKernel, FourierFeatures


class SparseGaussianProcessParameters(NamedTuple):
    log_error_stddev: jnp.ndarray
    inducing_locations: jnp.ndarray
    inducing_pseudo_mean: jnp.ndarray
    inducing_pseudo_log_err_stddev: jnp.ndarray
    kernel_params: NamedTuple


class SparseGaussianProcessState(NamedTuple):
    inducing_weights: jnp.ndarray
    cholesky: jnp.ndarray
    prior_state: NamedTuple


class SparseGaussianProcess:
    """A sparse Gaussian process, implemented as a Haiku module."""

    def __init__(
        self,
        kernel: AbstractKernel,
        num_inducing: int,
        num_basis: int,
        num_samples: int,
    ):
        """Initializes the sparse GP.

        Args:
            kernel: the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
            num_inducing: the number of inducing points per input dimension.
            num_basis: the number of prior basis functions.
            num_samples: the number of samples stored in the GP.
        """
        self.kernel = kernel
        self.prior = FourierFeatures(kernel, num_basis)
        self.input_dimension = kernel.input_dimension
        self.output_dimension = kernel.output_dimension
        self.num_inducing = num_inducing
        self.num_basis = num_basis
        self.num_samples = num_samples

    def init_params_with_state(
        self,
        key: jnp.ndarray,
    ) -> Tuple[SparseGaussianProcessParameters, SparseGaussianProcessState]:
        (k1, k2, k3) = jr.split(key, 3)
        kernel_params = self.kernel.init_params(k1)

        (S, OD, ID, M, L) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            self.num_inducing,
            self.num_basis,
        )
        log_error_stddev = jnp.zeros((OD))
        inducing_locations = jr.uniform(k2, (M, ID))
        inducing_pseudo_mean = jnp.zeros((M, OD))
        inducing_pseudo_log_err_stddev = jnp.zeros((M, OD))
        params = SparseGaussianProcessParameters(
            log_error_stddev,
            inducing_locations,
            inducing_pseudo_mean,
            inducing_pseudo_log_err_stddev,
            kernel_params,
        )

        inducing_weights = jnp.zeros((S, M, OD))
        cholesky = jnp.zeros((M, M, OD, OD))
        prior_state = self.prior.init_state(params.kernel_params, self.num_samples, k3)
        state = SparseGaussianProcessState(inducing_weights, cholesky, prior_state)

        return (params, state)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (S, OD, ID, M) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            self.num_inducing,
        )
        inducing_locations = params.inducing_locations
        kernel_params = params.kernel_params
        inducing_weights = state.inducing_weights
        prior_state = state.prior_state

        f_prior = self.prior(kernel_params, prior_state, x)
        K = self.kernel.matrix(kernel_params, x, inducing_locations)

        if len(x.shape) == 2:
            f_data = jnp.einsum("mnop,snp->smo", K, inducing_weights)  # non-batched
        elif len(x.shape) == 3:
            f_data = jnp.einsum("smnop,snp->smo", K, inducing_weights)  # non-batched

        return f_prior + f_data  # , (f_prior, f_data)

    # @partial(jax.jit, static_argnums=(0,))
    # def rollout(
    #     self,
    #     params: SparseGaussianProcessParameters,
    #     state: SparseGaussianProcessState,
    #     x: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     """Evaluates the sparse GP for a given input matrix.

    #     Args:
    #         x: the input matrix.
    #     """
    #     (S, OD, ID, M) = (
    #         self.num_samples,
    #         self.output_dimension,
    #         self.input_dimension,
    #         self.num_inducing,
    #     )
    #     inducing_locations = params.inducing_locations
    #     kernel_params = params.kernel_params
    #     inducing_weights = state.inducing_weights
    #     prior_state = state.prior_state

    #     f_prior = self.prior(kernel_params, prior_state, x)
    #     K = self.kernel.matrix(kernel_params, x, inducing_locations)
    #     f_data = jnp.einsum("smnop,snp->smo", K, inducing_weights)  # non-batched

    #     return f_prior + f_data  # , (f_prior, f_data)

    @partial(jax.jit, static_argnums=(0,))
    def randomize(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        key: jnp.ndarray,
    ) -> SparseGaussianProcessState:
        """Samples a new set of random functions from the GP."""
        (S, OD, ID, M, L) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            self.num_inducing,
            self.num_basis,
        )
        inducing_locations = params.inducing_locations
        inducing_pseudo_mean = params.inducing_pseudo_mean
        inducing_pseudo_log_err_stddev = params.inducing_pseudo_log_err_stddev
        kernel_params = params.kernel_params
        prior_state = state.prior_state

        inducing_pseudo_mean = rearrange(inducing_pseudo_mean, "M OD -> (M OD)")
        inducing_pseudo_log_err_stddev = rearrange(
            inducing_pseudo_log_err_stddev, "M OD -> (M OD)"
        )

        (k1, k2) = jr.split(key)
        prior_state = self.prior.resample_weights(
            kernel_params, prior_state, self.num_samples, k1
        )
        state = SparseGaussianProcessState(
            state.inducing_weights, state.cholesky, prior_state
        )

        K = self.kernel.matrix(kernel_params, inducing_locations, inducing_locations)
        M1, M2, OD1, OD2 = K.shape
        K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")

        inducing_noise_kernel = jnp.diag(jnp.exp(inducing_pseudo_log_err_stddev * 2))

        K = K + inducing_noise_kernel

        (cholesky, _) = jsp.linalg.cho_factor(
            K,
            lower=True,
        )

        prior = self.prior(params.kernel_params, state.prior_state, inducing_locations)
        prior = rearrange(prior, "S M OD -> S (M OD)")
        noise_cholesky = jnp.diag(jnp.exp(inducing_pseudo_log_err_stddev))
        sample_noise = jnp.einsum(
            "ij,sj->si", noise_cholesky, jr.normal(k2, (S, M * OD))
        )
        residual = prior + sample_noise

        inducing_weights = (
            inducing_pseudo_mean
            - tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(
                tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(
                    residual
                ),
                adjoint=True,
            )
        )  # mean-reparameterized v = \mu + (K + V)^{-1}(-f - \eps)

        cholesky = rearrange(
            cholesky,
            "(M1 OD1) (M2 OD2) -> M1 M2 OD1 OD2",
            M1=M1,
            M2=M2,
            OD1=OD1,
            OD2=OD2,
        )
        inducing_weights = rearrange(inducing_weights, "S (M OD) -> S M OD", M=M, OD=OD)

        return SparseGaussianProcessState(inducing_weights, cholesky, prior_state)

    @partial(jax.jit, static_argnums=(0,))
    def resample_prior_basis(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        key: jnp.ndarray,
    ):
        """Resamples the frequency and phase of the prior random feature basis."""
        prior_state = self.prior.resample_basis(
            params.kernel_params, state.prior_state, key
        )

        return SparseGaussianProcessState(
            state.inducing_weights, state.cholesky, prior_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def set_inducing_points(
        self,
        params: SparseGaussianProcessParameters,
        # state: SparseGaussianProcessState,
        inducing_locations: jnp.ndarray,
        inducing_means: jnp.ndarray,
        inducing_err_stddev: jnp.ndarray,
    ) -> SparseGaussianProcessParameters:
        # reparametrised mean

        M, OD = inducing_means.shape

        inducing_means = rearrange(inducing_means, "M OD -> (M OD)")
        inducing_err_stddev = rearrange(inducing_err_stddev, "M OD -> (M OD)")

        K = self.kernel.matrix(
            params.kernel_params, inducing_locations, inducing_locations
        )
        K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")

        inducing_noise_kernel = jnp.diag(jnp.power(inducing_err_stddev, 2))

        K = K + inducing_noise_kernel

        inducing_pseudo_mean = tf2jax.linalg.matvec(
            tf2jax.linalg.inv(K), inducing_means
        )

        # TODO: is this faster? gives the right answer
        # (cholesky, _) = jsp.linalg.cho_factor(
        #     K,
        #     lower=True,
        # )
        # inducing_pseudo_mean = tf2jax.linalg.LinearOperatorLowerTriangular(
        #     cholesky
        # ).solvevec(
        #     tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(
        #         inducing_means
        #     ),
        #     adjoint=True,
        # )

        inducing_pseudo_mean = rearrange(
            inducing_pseudo_mean, "(M OD) -> M OD", M=M, OD=OD
        )
        inducing_err_stddev = rearrange(
            inducing_err_stddev, "(M OD) -> M OD", M=M, OD=OD
        )

        params = params._replace(
            inducing_locations=inducing_locations,
            inducing_pseudo_mean=inducing_pseudo_mean,
            inducing_pseudo_log_err_stddev=jnp.log(inducing_err_stddev),
        )

        return params

    @partial(jax.jit, static_argnums=(0,))
    def get_inducing_mean(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
    ) -> jnp.ndarray:

        inducing_locations = params.inducing_locations
        inducing_pseudo_mean = params.inducing_pseudo_mean
        inducing_pseudo_log_err_stddev = params.inducing_pseudo_log_err_stddev

        M, OD = inducing_pseudo_mean.shape

        inducing_pseudo_mean = rearrange(inducing_pseudo_mean, "M OD -> (M OD)")
        inducing_err_stddev = jnp.exp(
            rearrange(inducing_pseudo_log_err_stddev, "M OD -> (M OD)")
        )

        K = self.kernel.matrix(
            params.kernel_params, inducing_locations, inducing_locations
        )
        K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")

        inducing_noise_kernel = jnp.diag(jnp.power(inducing_err_stddev, 2))

        K = K + inducing_noise_kernel

        inducing_mean = tf2jax.linalg.matvec(K, inducing_pseudo_mean)

        inducing_mean = rearrange(inducing_mean, "(M OD) -> M OD", M=M, OD=OD)

        return inducing_mean

    @partial(jax.jit, static_argnums=(0,))
    def prior_kl(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective."""
        (OD, ID, M) = (self.output_dimension, self.input_dimension, self.num_inducing)
        inducing_locations = params.inducing_locations
        inducing_pseudo_mean = params.inducing_pseudo_mean
        inducing_pseudo_log_err_stddev = params.inducing_pseudo_log_err_stddev
        kernel_params = params.kernel_params
        cholesky = state.cholesky

        cholesky = rearrange(cholesky, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")
        inducing_pseudo_mean = rearrange(inducing_pseudo_mean, "M OD -> (M OD)")
        inducing_pseudo_log_err_stddev = rearrange(
            inducing_pseudo_log_err_stddev, "M OD -> (M OD)"
        )

        logdet_term = 2 * jnp.sum(jnp.log(jnp.diag(cholesky))) - 2 * jnp.sum(
            inducing_pseudo_log_err_stddev
        )
        K = self.kernel.matrix(kernel_params, inducing_locations, inducing_locations)
        K = rearrange(K, "M1 M2 OD1 OD2 -> (M1 OD1) (M2 OD2)")
        cholesky_inv = tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(
            tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(
                jnp.eye(M * OD)
            ),
            adjoint=True,
        )
        trace_term = jnp.sum(cholesky_inv * K)
        reparameterized_quadratic_form_term = jnp.einsum(
            "i,ij,j", inducing_pseudo_mean, K, inducing_pseudo_mean
        )

        return (
            logdet_term - (OD * M) + trace_term + reparameterized_quadratic_form_term
        ) / 2

    @partial(jax.jit, static_argnums=(0,))
    def hyperprior(
        self, params: SparseGaussianProcessParameters, state: SparseGaussianProcessState
    ) -> jnp.ndarray:
        """Returns the log hyperprior regularization term of the GP."""
        return jnp.zeros(())  # temporary

    @partial(jax.jit, static_argnums=(0,))
    def loss(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        key: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, SparseGaussianProcessState]:

        state = self.randomize(params, state, key)

        kl = self.prior_kl(params, state)

        f = self(params, state, x)
        s = jnp.exp(params.log_error_stddev)
        (n_samples, n_batch, _) = f.shape
        c = n_data / (n_batch * n_samples * 2)
        l = n_data * jnp.sum(jnp.log(s)) + c * jnp.sum(((y - f) / s) ** 2)

        r = self.hyperprior(params, state)

        return (kl + l + r, state)

    @partial(jax.jit, static_argnums=(0,))
    def sample_parts(
        self,
        params: SparseGaussianProcessParameters,
        state: SparseGaussianProcessState,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (S, OD, ID, M) = (
            self.num_samples,
            self.output_dimension,
            self.input_dimension,
            self.num_inducing,
        )
        inducing_locations = params.inducing_locations
        kernel_params = params.kernel_params
        inducing_weights = state.inducing_weights
        prior_state = state.prior_state

        f_prior = self.prior(kernel_params, prior_state, x)
        K = self.kernel.matrix(kernel_params, x, inducing_locations)
        f_data = jnp.einsum("mnop,snp->smo", K, inducing_weights)  # non-batched

        return f_prior, f_data  # , (f_prior, f_data)
