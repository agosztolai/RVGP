
import torch
import gpytorch
import tqdm as tqdm
from math import floor
from gpytorch.kernels import InducingPointKernel, MultitaskKernel


X, y = torch.randn(1000, 3), torch.randn(1000)


train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()
train_y = torch.stack([train_y, train_y]).T

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()
test_y = torch.stack([test_y, test_y]).T

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.base_covar_module = MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :].clone(), likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = GPRegressionModel(train_x, train_y, likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
    
training_iterations = 10

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

iterator = tqdm.tqdm(range(training_iterations), desc="Train")

for i in iterator:
    # Zero backprop gradients
    optimizer.zero_grad()
    # Get output from model
    output = model(train_x)
    # Calc loss and backprop derivatives
    loss = -mll(output, train_y)
    loss.backward()
    iterator.set_postfix(loss=loss.item())
    optimizer.step()
    torch.cuda.empty_cache()
    
model.eval()
likelihood.eval()
with torch.no_grad():
    preds = model.likelihood(model(test_x))
    print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))
    print('Test NLL: {}'.format(-preds.to_data_independent_dist().log_prob(test_y).mean().item()))
        
