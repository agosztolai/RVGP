# RVGP - Riemannian manifold vector field Gaussian Processes

RVGP is a Generalised Gaussian process for learning vector fields over manifolds. It does so by generalising Gaussian processes using the connection Laplacian operator, which introduces smoothness in the tangent bundle of manifolds.

You will find RVGP useful if you want to 
1. Learn and interpolate vector fields from sparse samples
2. Infer the vector field in out-of-sample regions to recover the singularities
3. Globally smoothen noisy vector fields to better preserve singularities.

The package is based on [GPFlow 2.0](https://gpflow.github.io/GPflow/2.9.0/index.html).



https://github.com/agosztolai/RVGP/assets/34625112/744dd5d0-03b4-43ba-8720-d6993cacd076



## Cite

If you find this package useful or inspirational, please cite our work as follows

```
@misc{peach2023implicit,
      title={Implicit Gaussian process representation of vector fields over arbitrary latent manifolds}, 
      author={Robert L. Peach and Matteo Vinao-Carl and Nir Grossman and Michael David and Emma Mallas and David Sharp and Paresh A. Malhotra and Pierre Vandergheynst and Adam Gosztolai},
      year={2023},
      journal={ICLR}
}
```

## Installation

Create a new Anaconda environment. Then, install by running inside the main folder

```
pip install numpy cython scipy
pip install -e .
```

## Quick start

We suggest you study at least the example of a [superresolution of vector fields over manifolds](https://github.com/agosztolai/RVGP/blob/main/examples/surface_interpolation/superresolution_vector_field.py) to understand what behaviour to expect.

Briefly, RVGP takes the following inputs

1. `X` - an `nxd` array of points to define the points cloud, which are considered to be sampled from a smooth manifold.
2. `vectors` - an `nxd` array, defining a signal over the manifold.
3. (optional) explained_variance - This will be used to estimate the manifold dimension. You may want to change ```dim_man``` if the predicted dimension differs.

Before you fit RVGP, it is a good idea to perform furthest point sampling to even out sample points. This ensured that any particular region of the manifold would not be overfit. 

```
from RVGP.geometry import furthest_point_sampling
sample_ind, _ = furthest_point_sampling(X, stop_crit=0.015)
X = X[sample_ind]
```

Now, you are ready to create a data object. Specify the number of eigenvectors you wish to use. The following code will compute the necessary objects, including gauge fields, connections, connection Laplacian and eigendecomposition.

```
import RVGP
n_eigenpairs = 50
d = RVGP.create_data_object(X, vectors=vectors, n_eigenpairs=n_eigenpairs)
```

If you just want to play around and do not have a vector field, you can create one by sampling uniformly from the sphere

```
n_eigenpairs = 50
d = RVGP.create_data_object((X, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
```

You can then run vector diffusion to 'smooth it out'. The parameter ```t``` is the diffusion time, with a larger value resulting in smoother fields.

```
d.smooth_vector_field(t=100)
```

You are ready to train! The following code will train on 50% of the points.

```
train_ind =  np.random.choice(np.arange(len(X)), size=int(0.5*len(X)))
vector_field_GP = RVGP.fit(d, train_ind=train_ind, noise_variance=0.001)
```

Finally, test your trained GP on the remainder of the points.

```
test_ind = [i for i in range(len(X)) if i not in train_ind]
f_pred_mean, _ = vector_field_GP.transform(d, test_ind)
```

For ```test_ind```, you can either use integers, which will be interpreted as indices of nodes, or floats, which will be interpreted as positional encoding over the tangent bundle in the spectral domain.

We recommend using the [Polyscope](https://polyscope.run) package to perform beautiful visualisations. See examples for how to use it.

## Stay in touch

If you want to chat about your use case, get in touch or raise an issue! We are happy to help and looking to further develop this package to make it as useful as possible.

## References

The following packages were inspirational during the development of this code:

* [Vector heat method](https://github.com/nmwsharp/potpourri3d)
* [Polyscope](https://polyscope.run)
* [Parallel Transport Unfolding](https://github.com/mbudnins/parallel_transport_unfolding)
