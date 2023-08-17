import gpytorch
import torch
from torch import nn
from gpytorch.constraints import Positive


class ManifoldKernel(gpytorch.kernels.Kernel):
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

    def __init__(self, eigenpairs, nu=3, kappa=4, sigma_f=1, typ='matern', **kwargs):
        kwargs['batch_shape']=torch.Size([1])
        super(ManifoldKernel, self).__init__(**kwargs)
        
        self.eigenvectors, self.eigenvalues = eigenpairs
        self.num_verticies = self.eigenvectors.shape[0]
        
        self.typ = 'matern'
        
        # register the raw parameter
        self.register_parameter(
            name='nu', parameter=nn.Parameter(torch.tensor(float(nu)))
        )
        self.register_parameter(
            name='kappa', parameter=nn.Parameter(torch.tensor(float(kappa)))
        )
        self.register_parameter(
            name='sigma_f', parameter=nn.Parameter(torch.tensor(float(sigma_f)))
        )
        
        # set the parameter constraint to be positive, when nothing is specified
        length_constraint = Positive()

        # register the constraint
        self.register_constraint("nu", length_constraint)
        self.register_constraint("kappa", length_constraint)
        self.register_constraint("sigma_f", length_constraint)
        
    def forward(self, x1, x2=None, **params):
        """Kernel function"""
        
        if x2 is None:
            x2 = x1
            
        x1, x2 = x1.squeeze(), x2.squeeze()
        S = self.eval_S()
        covar = (x1 * S) @ x2.t() # shape (n,n)
        
        return covar.unsqueeze(0) # shape (1,n,n)
        

    def eval_S(self, typ = 'matern'):
        """Wilson Eq. (69)"""
        if self.typ == 'matern':
            S = torch.pow(self.eigenvalues + 2*self.nu/self.kappa**2, -self.nu)
            S = torch.multiply(S, self.num_verticies/S.sum())
            S = torch.multiply(S, self.sigma_f)
            
        elif self.typ == 'SE':
            S = torch.exp(-0.5*self.eigenvalues*self.kappa)
            S = torch.multiply(S, self.num_verticies/S.sum())
            S = torch.multiply(S, self.sigma_f)
        
        return S