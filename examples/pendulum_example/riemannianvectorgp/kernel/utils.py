import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial


def pairwise_dimension_distance(x1, x2):
    return x1[..., np.newaxis, :] - x2[..., np.newaxis, :, :]
