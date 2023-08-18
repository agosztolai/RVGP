from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels


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
    def kernel(
        self,
        params: NamedTuple,
    ):
        pass

    @abstractmethod
    def standard_spectral_measure(
        self, key: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def spectral_weights(
        self,
        params: NamedTuple,
        frequency: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass


class ScaledKernelParameters(NamedTuple):
    log_amplitudes: jnp.ndarray
    log_length_scale: jnp.ndarray


class ScaledKernel(AbstractKernel):
    """A kernel with learned amplitude and length scale parameters."""

    def __init__(
        self,
        kernel_class: ABCMeta,
        input_dimension: int,
        output_dimension: int,
    ):
        """Scales the given kernel input by length scales and output by amplitudes.

        Args:
            kernel_class: the class of the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
        """
        super().__init__()
        self.kernel_class = kernel_class
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def init_params(
        self,
        key: jnp.ndarray,
    ) -> ScaledKernelParameters:
        log_amplitudes = jnp.zeros((self.output_dimension))
        log_length_scale = jnp.zeros((self.input_dimension))
        return ScaledKernelParameters(log_amplitudes, log_length_scale)

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: ScaledKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """Assemble the kernel matrix.

        Args:
            x1: the first input.
            x2: the second input.
        """
        return self.kernel(params).matrix(x1, x2)

    def kernel(
        self,
        params: ScaledKernelParameters,
    ):
        """Instantiates the kernel with the given parameters."""
        amplitudes = jnp.exp(params.log_amplitudes)
        length_scales = jnp.exp(params.log_length_scale)
        return self.kernel_class(amplitude=amplitudes, length_scale=length_scales)

    @partial(jit, static_argnums=(0, 2))
    def standard_spectral_measure(
        self, key: jnp.ndarray, num_samples: int
    ) -> jnp.ndarray:
        """Draws samples from the kernel's spectral measure.

        Args:
            num_samples: the number of samples to draw.
        """
        if self.kernel_class == tfk.ExponentiatedQuadratic:
            return jr.normal(
                key, (self.output_dimension, self.input_dimension, num_samples)
            )
        else:
            raise Exception("Spectral measure not implemented for this kernel.")

    @partial(jit, static_argnums=(0,))
    def spectral_weights(
        self,
        params: ScaledKernelParameters,
        frequency: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the input weights and output weights associated with the kernel.

        Args:
            kernel: the kernel.
            frequency: the sampled frequencies.
        """
        amplitudes = jnp.exp(params.log_amplitudes)
        length_scales = jnp.exp(params.log_length_scale)
        return (amplitudes, length_scales)
