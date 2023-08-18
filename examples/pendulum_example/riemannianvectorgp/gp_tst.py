import unittest
from riemannianvectorgp.gp import SparseGaussianProcess
from sparsegpax import gp
import spectral
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
import jax
import numpy as np
import jax.random as jr
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized


class GPTest(parameterized.TestCase):
    # This uses the old API for spectral which does not handle multi output
    @absltest.skip
    def test_prior_basis(self):
        kernel = tfk.ExponentiatedQuadratic(length_scale=1.0, amplitude=1.0)
        key = jax.random.PRNGKey(0)
        input_dim = 1
        output_dim = 1
        num_basis = 1024
        num_train_points = 4
        num_samples = 1  # FIXME
        key_x, key_freq, key_phase = jax.random.split(key, 3)
        x = jax.random.uniform(key_x, (num_train_points, input_dim))
        # w = spectral.standard_spectral_measure(kernel, input_dim, num_samples, key)
        # prior_frequency has shape (num_basis, 1, input_dim)
        frequency = spectral.standard_spectral_measure(
            kernel, input_dim, num_basis, key_freq
        )
        frequency = jnp.squeeze(frequency, axis=-1)  # (num_basis, input_dim)
        phase = jr.uniform(key_phase, (num_basis,), maxval=2 * jnp.pi)
        phi = lambda x: jnp.sqrt(2 / num_basis) * jnp.cos(x @ frequency.T + phase)
        # phi(x) # compute the feature_map, (n_train, num_basis)
        # print(phi(x).shape)
        # jnp.sum(phi(x) * phi(x))
        # mc_cov = jnp.cov(phi(x), bias=True)
        mc_cov = phi(x) @ phi(x).T
        analytical_cov = kernel.matrix(x, x)
        print(mc_cov.shape)
        print(analytical_cov.shape)
        print("Monte Carlo")
        print(mc_cov)
        print("Analytic")
        print(analytical_cov)
        print(jnp.abs(mc_cov - analytical_cov) / analytical_cov)
        print(jnp.max(jnp.abs(mc_cov - analytical_cov) / analytical_cov))

    @parameterized.parameters(
        {"input_dim": 3, "length_scale": 1.0, "seed": 1, "amplitude": None},
        # FIXME(ethan): non-default amplitude fails
        # {'input_dim': 2, 'length_scale': 1., 'seed': 0, 'amplitude': 1.},
    )
    def test_prior_sample_matches_analytical(
        self, input_dim, length_scale, seed, amplitude
    ):
        output_dim = 2
        # The current test only handles output_dim == 1
        # assert output_dim == 1
        num_basis = 4096
        num_train_points = 5
        num_function_samples = 5000
        length_scales = jnp.ones((input_dim,)) * length_scale
        # FIXME(ethan): the following should pass
        # length_scales = jnp.ones((output_dim, input_dim)) * length_scale
        num_inducing_points = 16

        kernel = tfk.FeatureScaled(
            tfk.ExponentiatedQuadratic(amplitude=amplitude),
            scale_diag=length_scales ** 2,  # Use FeatureScaled for RBF ARD
        )

        key = jax.random.PRNGKey(seed)

        model = gp.SparseGaussianProcess(
            input_dimension=input_dim,
            output_dimension=output_dim,
            kernel=kernel,
            key=key,
            num_basis=num_basis,
            num_samples=num_function_samples,
            num_inducing=num_inducing_points,
        )

        x = jax.random.uniform(key, (num_train_points, input_dim))

        # Generate samples at x
        f = model.prior(x)
        self.assertEqual(f.shape, (num_function_samples, num_train_points, output_dim))
        # f = jnp.squeeze(model.prior(x), -1) # assuming output dimension is 1
        analytical_mean = jnp.zeros((num_train_points, output_dim))
        analytical_cov = kernel.matrix(x, x)

        sample_mean = jnp.mean(f, axis=0)
        cov_fn = lambda x: jnp.cov(x, rowvar=False, bias=True)
        sample_cov = jax.vmap(cov_fn, in_axes=2)(f)

        self.assertLessEqual(
            jnp.max(jnp.abs(sample_cov - analytical_cov) / analytical_cov), 0.2
        )

    def test_posterior_sample_shape(self):
        input_dim = 2
        output_dim = 1
        num_basis = 4096
        num_train_points = 3
        num_function_samples = 5000
        length_scales = jnp.ones((input_dim,))
        num_inducing_points = 16

        kernel = tfk.FeatureScaled(
            tfk.ExponentiatedQuadratic(), scale_diag=length_scales ** 2
        )

        key = jax.random.PRNGKey(0)

        model = gp.SparseGaussianProcess(
            input_dimension=input_dim,
            output_dimension=output_dim,
            kernel=kernel,
            key=key,
            num_basis=num_basis,
            num_samples=num_function_samples,
            num_inducing=num_inducing_points,
        )
        num_test_points = 7
        xtest = jax.random.uniform(key, (num_test_points, input_dim))
        self.assertEqual(
            model(xtest).shape, (num_function_samples, num_test_points, output_dim)
        )


if __name__ == "__main__":
    absltest.main()
