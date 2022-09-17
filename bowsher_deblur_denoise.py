"""example script for MR reconstruction with smooth cost function using CG and LBFGS"""
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from scipy.ndimage import gaussian_filter

from operators import BowsherGradient1D, Convolution1D
from functionals import SmoothNorm, TotalCost, L2NormSquared

#----------------------------------------------------------------------------------------
# input parameters

n: int = 256
noise_level: float = 0.2
seed: int = 0

num_iter: int = 200

data_norm: SmoothNorm = L2NormSquared()
prior_norm: SmoothNorm = L2NormSquared()

# generate a Gaussian kernel, the last argument of np.linspace can be used to change its width
kernel: npt.NDArray = np.exp(-np.linspace(-2.5, 2.5, 23)**2)

neighbors: tuple = (-3, -2, 1, 1, 2, 3)
num_nearest: int = 2

betas: npt.NDArray = np.array([0, 0.1, 1, 10])
#----------------------------------------------------------------------------------------
np.random.seed(seed)

# normalize the kernel
kernel /= kernel.sum()

# setup ground truth step like signal
x_true = gaussian_filter(np.random.rand(n), 10)
x_true -= x_true.min()
x_true *= (3 / x_true.max())
x_true = np.round(x_true)

x_true[:15] = 0
x_true[-15:] = 0

x_true /= x_true.max()

# setup the prior signal
prior_signal = (x_true.max() - x_true) * np.cos(3 * x_true)
prior_signal[(n // 2):] *= -1

# add mismatch in prior signal
prior_signal[180:185] = prior_signal.max()
prior_signal[190:195] = prior_signal.max()
prior_signal[200:205] = prior_signal.max()

# add stand alone feature in true signal
x_true[65:75] = 0.5

# setup the data model
data_operator = Convolution1D(n, kernel)

# generate noise-free data
noise_free_data = data_operator.forward(x_true)

noisy_data = noise_free_data + noise_level * x_true.mean() * np.random.randn(
    *data_operator.x_shape)

# apply the adjoint of the data operator to the data
data_back = data_operator.adjoint(noisy_data)

# setup the prior operator using anatomical information (limited number of nearest neighbors)
structural_prior_operator = BowsherGradient1D(n, neighbors, num_nearest,
                                              prior_signal)

# setup the prior operator using anatomical information (limited number of nearest neighbors)
non_structural_prior_operator = BowsherGradient1D(n, neighbors, len(neighbors),
                                                  prior_signal)

#----------------------------------------------------------------------------------------
# run the recons

recons_str = np.zeros((betas.size, n))
recons_non_str = np.zeros((betas.size, n))

for i, beta in enumerate(betas):
    # the total cost function that has __call__(x) and gradient(x)
    cost_str = TotalCost(noisy_data, data_operator, data_norm,
                         structural_prior_operator, prior_norm,
                         beta / structural_prior_operator.num_nearest)

    cost_non_str = TotalCost(noisy_data, data_operator, data_norm,
                             non_structural_prior_operator, prior_norm,
                             beta / non_structural_prior_operator.num_nearest)

    # initial recon
    x0 = np.zeros(data_operator.x_shape).ravel()

    # recon using conjugate gradient optimizer
    res_str = fmin_cg(cost_str,
                      x0=x0,
                      fprime=cost_str.gradient,
                      maxiter=num_iter,
                      full_output=True)

    recons_str[i, :] = res_str[0].reshape(data_operator.x_shape)

    # recon using conjugate gradient optimizer
    res_non_str = fmin_cg(cost_non_str,
                          x0=x0,
                          fprime=cost_non_str.gradient,
                          maxiter=num_iter,
                          full_output=True)

    recons_non_str[i, :] = res_non_str[0].reshape(data_operator.x_shape)

#--------------------------------------------------------------------------------
# show the results
fig, ax = plt.subplots(2,
                       betas.size,
                       figsize=(betas.size * 4, 2 * 4),
                       sharex=True,
                       sharey=True)
for i, beta in enumerate(betas):
    ax[0, i].fill_between(np.arange(n),
                          x_true,
                          0 * x_true + x_true.min(),
                          color=(0.8, 0.8, 0.8))
    ax[0, i].plot(np.arange(n),
                  noisy_data,
                  color='tab:blue',
                  label='blurry and noisy signal')
    ax[0, i].plot(np.arange(n),
                  recons_non_str[i, :],
                  color='tab:orange',
                  label='it. recon')

    ax[1, i].fill_between(np.arange(n),
                          x_true,
                          0 * x_true + x_true.min(),
                          color=(0.8, 0.8, 0.8))
    ax[1, i].plot(np.arange(n),
                  noisy_data,
                  color='tab:blue',
                  label='blurry and noisy signal')
    ax[1, i].plot(np.arange(n),
                  recons_str[i, :],
                  color='tab:red',
                  label='it. recon')

    ax[0, i].set_title(f'prior weight {beta}', fontsize='medium')

ax[0, -1].legend()
ax[1, -1].legend()
ax[0, 0].set_ylabel('non structural prior')
ax[1, 0].set_ylabel('structural prior')

for axx in ax.ravel():
    axx.set_ylim(-0.2, 1.3 * x_true.max())
    axx.grid(ls=':')

fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots(figsize=(4, 4))
ax2.fill_between(np.arange(n),
                 x_true,
                 0 * x_true + x_true.min(),
                 color=(0.8, 0.8, 0.8),
                 label='true signal')
ax2.plot(prior_signal, 'k', label='structural signal')
ax2.legend()
ax2.grid(ls=':')
fig2.tight_layout()
fig2.show()
