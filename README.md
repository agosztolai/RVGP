# RVGP - Riemannian manifold vector field Gaussian Processes

RVGP is a Generalised Gaussian process for learning vector fields over manifolds. It does so by generalising Gaussian processes using the connection Laplacian operator, which introduces smoothness in the tangent bundle of manifolds.

You will find RVGP useful if you want to 
1. Learn and interpolate vector fields from sparse samples
2. Infer the vector field in out-of-sample regions in order to recover the singularities
3. Globally smoothen noisy vector fields in order to better preserve singularities.

The package is based on [GPFlow 2.0](https://gpflow.github.io/GPflow/2.9.0/index.html).

## Cite

If you find this package useful or inspirational, please cite our work as follows

```
@misc{Peach2023,
      title={Interpretable statistical representations of neural population dynamics and geometry}, 
      author={Robert L. Peach, Matteo Vinao-Carl, Nir Grossman, Michael David, Emma Mallas, David J. Sharp, Paresh A. Malhotra, Pierre Vandergheynst and Adam Gosztolai},
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Installation

Create a new Anaconda environment. Then, install by running inside the main folder

```
pip install numpy, cython
pip install -e .
```

## Quick start

We suggest you study at least the example of a [superresolution of vector fields over manifolds](https://github.com/agosztolai/RVGP/blob/main/examples/surface_interpolation/superresolution_vector_field.py) to understand what behaviour to expect.

Briefly, RVGP takes the following inputs

1. `X` - an `nxd` array of points to define the points cloud, which are considered to be sampled from a smooth manifold.
2. `vectors` - an `nxd` array, defining a signal over the manifold.
3. dim_man - dimension of the manifold.
4. (optional) explained_variance - This will be used to estimate the dimension of the manifold. You may want to change ```dim_man``` if the predicted dimension is different.

Before you fit RVGP, it is a good idea to perform furthest point sampling to even out sample points. This ensured that one region of the manifold will not be overfit. 

```
from RVGP.geometry import furthest_point_sampling

sample_ind, _ = furthest_point_sampling(X, stop_crit=0.015)
X = X[sample_ind]
```

Now you are ready to create a data object. Specify the number of eigenvectors you wish to use. Running the following code will compute the necessary objects including gauge fields, connections, connection Laplacian and eigendecomposition.

```
from RVGP import data
n_eigenpairs = 50
d = data(X, vectors=vectors, n_eigenpairs=n_eigenpairs)
```

If you just want to play around and do not have a vector field, you can create one by sampling uniformly from the sphere

```
n_eigenpairs = 50
d = data(X, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
```

You can then run vector diffusion on it to 'smooth it out'. The parameter ```t``` is the diffusion time, with larger value resulting in smoother fields.

```
d.smooth_vector_field(t=100)
```

You are ready to train! First, define a kernel using the eigenvectors of the connection Laplacian operator. You can leave the parameters as is, as they are just initial conditions for the optimiser. You can also just leave the ```kernel``` argument empty to use radial basis functions or specify your favourite kernel from the GPFlow 2.0 library. 

```
from RVGP.kernels import ManifoldKernel

vector_field_kernel = ManifoldKernel((d.evecs_Lc, d.evals_Lc), 
                                     nu=3/2, 
                                     kappa=5, 
                                     typ='matern',
                                     sigma_f=1.)
```

Go train! If you are uncertain about your data, you can set the noise_variance a bit higher. The ```dim``` argument is just the dimension of the ambient space.

```
from RVGP import train_gp
train_x, train_f = d.evecs_Lc.reshape(d.n, -1), d.vectors
vector_field_GP = train_gp(train_x,
                           train_f,
                           dim=vertices.shape[1],
                           kernel=vector_field_kernel,
                           noise_variance=0.001)
```

Finally, test your trained GP

```
test_x = d.evecs_Lc.reshape(d.n, -1)
n = len(test_x)
test_x = test_x.reshape(-1, n_eigenpairs)
f_pred_mean, _ = vector_field_GP.predict_f(test_x)
f_pred_mean = f_pred_mean.numpy().reshape(n, -1)
```

We recommend using the [Polyscope](https://polyscope.run) package to perform beautiful visualisations. See examples for how to use it.

## Stay in touch

If you want to chat about your use case, get in touch or raise an issue! We are happy to help and looking to further develop this package to make it as useful as possible.

## References

The following packages were inspirational during the development of this code:

* [Vector heat method](https://github.com/nmwsharp/potpourri3d)
* [Polyscope](https://polyscope.run)
* [Parallel Transport Unfolding](https://github.com/mbudnins/parallel_transport_unfolding)
