from abc import ABC, ABCMeta, abstractmethod
from typing import NamedTuple, Tuple
from functools import reduce
import operator
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

# import tensorflow_probability
# from tensorflow_probability.python.internal.backend import jax as tf2jax
from einops import rearrange

from .kernel import AbstractKLKernel, AbstractRFFKernel, AbstractKernel
from .utils import pairwise_dimension_distance

# tfp = tensorflow_probability.experimental.substrates.jax
# tfk = tfp.math.psd_kernels


class ProductKernelParams(NamedTuple):
    sub_kernel_params: list


class ProductFeatureState(NamedTuple):
    sub_feature_states: list


class ProductKernel(AbstractKernel):
    def __init__(self, *subkernels, rff_kl_ratio=1.0):
        self.sub_kernels = subkernels
        self.KL_indicies = [
            i for i, k in enumerate(self.sub_kernels) if isinstance(k, AbstractKLKernel)
        ]
        self.RFF_indicies = [
            i
            for i, k in enumerate(self.sub_kernels)
            if isinstance(k, AbstractRFFKernel)
        ]
        self.sub_input_dimensions = [k.input_dimension for k in self.sub_kernels]
        self.sub_output_dimensions = [k.output_dimension for k in self.sub_kernels]

        self.input_dimension = sum(self.sub_input_dimensions)
        self.output_dimension = max(self.sub_output_dimensions)

        self.rff_kl_ratio = rff_kl_ratio

        # Check at least shapewise the product is valid.
        for od in self.sub_output_dimensions:
            assert (od == self.output_dimension) or (od == 1)

    def init_params(
        self,
        key: jnp.ndarray,
    ) -> ProductKernelParams:
        sub_kernel_params = []

        for sub_kernel in self.sub_kernels:
            key, part = jr.split(key)
            sub_kernel_params.append(sub_kernel.init_params(part))

        return ProductKernelParams(sub_kernel_params)

    @partial(jit, static_argnums=(0,))
    def matrix(
        self,
        params: NamedTuple,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:

        sub_x1 = jnp.split(x1, self.sub_input_dimensions[:-1], axis=-1)
        sub_x2 = jnp.split(x2, self.sub_input_dimensions[:-1], axis=-1)

        sub_matricies = [
            k.matrix(sub_params, x1, x2)
            for k, x1, x2, sub_params in zip(
                self.sub_kernels, sub_x1, sub_x2, params.sub_kernel_params
            )
        ]

        return reduce(operator.mul, sub_matricies)

    @partial(jit, static_argnums=(0, 3))
    def sample_fourier_features(
        self,
        params: NamedTuple,
        key: jnp.ndarray,
        num_samples: int,
    ) -> NamedTuple:
        sub_feature_states = []

        for i, (sub_kernel, sub_params) in enumerate(
            zip(self.sub_kernels, params.sub_kernel_params)
        ):
            key, part = jr.split(key)
            if i in self.KL_indicies:
                sub_num_samples = num_samples
            else:
                sub_num_samples = int(self.rff_kl_ratio * num_samples)
            sub_feature_states.append(
                sub_kernel.sample_fourier_features(sub_params, part, sub_num_samples)
            )

        return ProductFeatureState(sub_feature_states)

    @partial(jit, static_argnums=(0,))
    def weight_variance(
        self,
        params: NamedTuple,
        state: NamedTuple,
    ) -> jnp.ndarray:

        if (len(self.KL_indicies) > 0) and (len(self.RFF_indicies) > 0):
            KL_sub_weight_variances = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].weight_variance(
                        params.sub_kernel_params[i], state.sub_feature_states[i]
                    )
                    for i in self.KL_indicies
                ],
            )
            RFF_sub_weight_variances = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].weight_variance(
                        params.sub_kernel_params[i], state.sub_feature_states[i]
                    )
                    for i in self.RFF_indicies
                ],
            )
            sub_weight_variances = (
                KL_sub_weight_variances[..., :, np.newaxis, :]
                * RFF_sub_weight_variances[..., np.newaxis, :, :]
            )
            # print(f"{KL_sub_weight_variances.shape=}")
            # print(f"{RFF_sub_weight_variances.shape=}")
            # print(f"{sub_weight_variances.shape=}")
            sub_weight_variances = rearrange(
                sub_weight_variances, "... M N O -> ... (M N) O"
            )
            # print(f"{sub_weight_variances.shape=}")
        elif len(self.KL_indicies) > 0:
            sub_weight_variances = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].weight_variance(
                        params.sub_kernel_params[i], state.sub_feature_states[i]
                    )
                    for i in self.KL_indicies
                ],
            )
        elif len(self.RFF_indicies) > 0:
            sub_weight_variances = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].weight_variance(
                        params.sub_kernel_params[i], state.sub_feature_states[i]
                    )
                    for i in self.RFF_indicies
                ],
            )
        else:
            raise ValueError("There appear to be no kernels in this product kernel")

        return sub_weight_variances

    @partial(jit, static_argnums=(0,))
    def basis_functions(
        self,
        params: NamedTuple,
        state: NamedTuple,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        sub_x = jnp.split(x, self.sub_input_dimensions[:-1], axis=-1)

        if (len(self.KL_indicies) > 0) and (len(self.RFF_indicies) > 0):
            KL_basis_functions = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].basis_functions(
                        params.sub_kernel_params[i],
                        state.sub_feature_states[i],
                        sub_x[i],
                    )
                    for i in self.KL_indicies
                ],
            )
            RFF_basis_functions = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].basis_functions(
                        params.sub_kernel_params[i],
                        state.sub_feature_states[i],
                        sub_x[i],
                    )
                    for i in self.RFF_indicies
                ],
            )
            basis_functions = (
                KL_basis_functions[..., :, np.newaxis, :, :]
                * RFF_basis_functions[..., np.newaxis, :, :, :]
            )
            basis_functions = rearrange(basis_functions, "... M N O E-> ... (M N) O E")
        elif len(self.KL_indicies) > 0:
            basis_functions = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].basis_functions(
                        params.sub_kernel_params[i],
                        state.sub_feature_states[i],
                        sub_x[i],
                    )
                    for i in self.KL_indicies
                ],
            )
        elif len(self.RFF_indicies) > 0:
            basis_functions = reduce(
                operator.mul,
                [
                    self.sub_kernels[i].basis_functions(
                        params.sub_kernel_params[i],
                        state.sub_feature_states[i],
                        sub_x[i],
                    )
                    for i in self.RFF_indicies
                ],
            )
        else:
            raise ValueError("There appear to be no kernels in this product kernel")

        return basis_functions
