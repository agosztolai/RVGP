import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def circle_distance(x1, x2):
    diff = jnp.abs(x1 - x2)
    return jnp.min(jnp.stack([diff, 2 * jnp.pi - diff], axis=-1), axis=-1)
