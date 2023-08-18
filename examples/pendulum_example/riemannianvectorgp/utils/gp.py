import jax
import jax.numpy as jnp
import optax


def train_sparse_gp(
    gp,
    gp_params,
    gp_state,
    m_cond,
    v_cond,
    rng,
    epochs=300,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    lr=0.01,
):
    opt = optax.chain(optax.scale_by_adam(b1=b1, b2=b2, eps=eps), optax.scale(-lr))
    opt_state = opt.init(gp_params)
    debug_params = [gp_params]
    debug_states = [gp_state]
    debug_keys = [rng.key]
    losses = []
    for i in range(epochs):
        ((train_loss, gp_state), grads) = jax.value_and_grad(gp.loss, has_aux=True)(
            gp_params, gp_state, next(rng), m_cond, v_cond, m_cond.shape[0]
        )
        (updates, opt_state) = opt.update(grads, opt_state)
        gp_params = optax.apply_updates(gp_params, updates)
        # if jnp.all(jnp.isnan(grads.kernel_params.sub_kernel_params.log_length_scale)):
        #     print("breaking for nan")
        #     break
        if i <= 10 or i % 20 == 0:
            print(i, "Loss:", train_loss)
        losses.append(train_loss)
        debug_params.append(gp_params)
        debug_states.append(gp_state)
        debug_keys.append(rng.key)

    return gp_params, gp_state, (debug_params, debug_states, debug_keys, losses)


def normalise_scaled_kernel(kernel, kernel_params, locations):
    kernel_params = kernel_params._replace(
        log_amplitude=-jnp.log(
            kernel.matrix(kernel_params, locations, locations)[0, 0, 0, 0]
        )
    )
    return kernel_params
