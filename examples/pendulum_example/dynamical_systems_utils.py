

import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels





def zero_grad_named_tuple(named_tuple):
    fields = []
    for sub in named_tuple:
        if isinstance(sub, jnp.DeviceArray):
            fields.append(jnp.zeros_like(sub))
        else:
            fields.append(zero_grad_named_tuple(sub))
    if isinstance(named_tuple, list):
        return fields
    else:
        return named_tuple.__class__(*fields)


def reverse_eular_integrate_rollouts(
    rollouts, system, estimate_momentum=False, thinning=1, chuck_factor=10
):

    if estimate_momentum:
        positions = rollouts[..., :, 0]
        momentums = (
            (positions[..., 1:] - positions[..., :-1] / system.step_size)
            * system.length
            * system.mass
            / 2
        )
        positions = positions[..., :-1]
        rollouts = jnp.stack([positions, momentums], axis=-1)

    deltas = (rollouts[..., 1:, :] - rollouts[..., :-1, :]) / system.step_size
    rollouts = rollouts[..., :-1, :]

    deltas = deltas[..., ::thinning, :]
    rollouts = rollouts[..., ::thinning, :]

    rollouts = rollouts.reshape((-1, rollouts.shape[-1]))
    deltas = deltas.reshape((-1, rollouts.shape[-1]))
    # Sketchy, chuckout the big delats that are fake...
    delta_norm = jnp.linalg.norm(deltas, axis=-1)
    delta_mean = jnp.mean(jnp.linalg.norm(deltas, axis=-1))

    chuck_inds = delta_norm > delta_mean * chuck_factor

    return rollouts[~chuck_inds], deltas[~chuck_inds]

