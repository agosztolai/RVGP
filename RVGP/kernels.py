# import numpy as np
import tensorflow as tf
import gpflow
import warnings


class ManifoldKernel(gpflow.kernels.Kernel):
    """Matern kernel on manifold. 

    Attributes
    ----------
    eigenpairs : tuple
        Truncated tuple returned by tf.linalg.eigh applied to the Laplacian of the graph.
    typ : str
        Type of kernel. 'matern' or 'SE'
    nu : float
        Trainable smoothness hyperparameter.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    dtype : tf.dtypes.DType
        type of tensors, tf.float64 by default
        """

    def __init__(self, eigenpairs, nu=3, kappa=4, sigma_f=1, dtype=tf.float64):

        self.eigenvectors, self.eigenvalues = eigenpairs
        self.num_verticies = tf.cast(tf.shape(self.eigenvectors)[0], dtype=dtype)
        self.dtype = dtype

        self.nu = gpflow.Parameter(nu, dtype=self.dtype, transform=gpflow.utilities.positive(), name='nu')
        self.kappa = gpflow.Parameter(kappa, dtype=self.dtype, transform=gpflow.utilities.positive(), name='kappa')
        self.sigma_f = gpflow.Parameter(sigma_f, dtype=self.dtype, transform=gpflow.utilities.positive(), name='sigma_f')
        super().__init__()

    def eval_S(self, typ = 'matern'):
        """Wilson Eq. (69)"""
        if typ == 'matern':
            S = tf.pow(self.eigenvalues + 2*self.nu/self.kappa**2, -self.nu)
            S = tf.multiply(S, self.num_verticies/tf.reduce_sum(S))
            S = tf.multiply(S, self.sigma_f)
            
        elif typ == 'SE':
            S = tf.exp(-0.5*self.eigenvalues*self.kappa)
            S = tf.multiply(S, self.num_verticies/tf.reduce_sum(S))
            S = tf.multiply(S, self.sigma_f)
        
        return S

    def K(self, X, X2=None):
        """Kernel function"""
        if X2 is None:
            X2 = X
            
        S = self.eval_S()
        return (X * S) @ tf.transpose(X2) # shape (n,n)

    def K_diag(self, X):
        """This is just the diagonal of K"""
        
        S = self.eval_S()
        return tf.reduce_sum(tf.transpose(X * S) * tf.transpose(X), axis=0)

    # def sample(self, X):
    #     K_chol = tf_jitchol(self.K(X), dtype=self.dtype)
    #     sample = K_chol.dot(np.random.randn(tf.shape(K_chol)[0]))
        
    #     return sample
    
    
# def tf_jitchol(mat, jitter=0, dtype=tf.float32):
#     """Run Cholesky decomposition with an increasing jitter,
#     until the jitter becomes too large.
#     Arguments
#     ---------
#     mat : (m, m) tf.Tensor
#         Positive-definite matrix
#     jitter : float
#         Initial jitter
#     """
#     try:
#         chol = tf.linalg.cholesky(mat)
        
#         return chol
    
#     except:
#         new_jitter = jitter*10.0 if jitter > 0.0 else 1e-15
#         if new_jitter > 1.0:
#             raise RuntimeError('Matrix not positive definite even with jitter')
#         warnings.warn(
#             'Matrix not positive-definite, adding jitter {:e}'
#             .format(new_jitter),
#             RuntimeWarning)
#         new_jitter = tf.cast(new_jitter, dtype=dtype)
        
#         return tf_jitchol(mat + tf.multiply(new_jitter, tf.eye(mat.shape[-1], dtype=dtype)), new_jitter)

