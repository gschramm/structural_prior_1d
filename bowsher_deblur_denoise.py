"""example script for MR reconstruction with smooth cost function using CG and LBFGS"""

import argparse
from ast import arg
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from scipy.ndimage import gaussian_filter

from operators import BowsherGradient1D, Convolution1D
from functionals import SmoothNorm, TotalCost, L2NormSquared

#----------------------------------------------------------------------------------------
# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--noise_level', default=0.15, type=float)
parser.add_argument('--n', default=256, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--max_iter', default=500, type=int)
parser.add_argument('--betas',
                    default=[0, 0.01, 0.1, 1, 10, 100],
                    type=float,
                    nargs='+')
parser.add_argument('--neighbors',
                    default=[-3, -2, -1, 1, 2, 3],
                    type=int,
                    nargs='+')
parser.add_argument('--num_nearest', default=2, type=int)
parser.add_argument('--sigma', default=1., type=float)
parser.add_argument('--sigma_mismatched', default=0.7, type=float)
args = parser.parse_args()
print(args)

noise_level: float = args.noise_level
n: int = args.n
seed: int = args.seed
num_iter: int = args.max_iter
betas: npt.NDArray = np.array(args.betas)
neighbors: tuple = tuple(args.neighbors)
num_nearest: int = args.num_nearest
sigma: float = args.sigma
sigma_mismatched: float = args.sigma_mismatched

data_norm: SmoothNorm = L2NormSquared()
prior_norm: SmoothNorm = L2NormSquared()

# generate a Gaussian kernel, the last argument of np.linspace can be used to change its width
tmp = np.linspace(-3, 3, 23)
kernel: npt.NDArray = np.exp(-(tmp**2) / (2 * sigma**2))
kernel_mismatched: npt.NDArray = np.exp(-(tmp**2) / (2 * sigma_mismatched**2))

#----------------------------------------------------------------------------------------
np.random.seed(seed)

# normalize the kernel
kernel /= kernel.sum()
kernel_mismatched /= kernel_mismatched.sum()

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

#----------------------------------------------------------------------------------------
# run the recons
data_operator_mismatched = Convolution1D(n, kernel_mismatched)

# apply the adjoint of the data operator to the data
data_back = data_operator_mismatched.adjoint(noisy_data)

# setup the prior operator using anatomical information (limited number of nearest neighbors)
structural_prior_operator = BowsherGradient1D(n, neighbors, num_nearest,
                                              prior_signal)

# setup the prior operator using anatomical information (limited number of nearest neighbors)
non_structural_prior_operator = BowsherGradient1D(n, neighbors, len(neighbors),
                                                  prior_signal)

recons_str_mismatched = np.zeros((betas.size, n))
recons_non_str_mismatched = np.zeros((betas.size, n))
recons_str_matched = np.zeros((betas.size, n))
recons_non_str_matched = np.zeros((betas.size, n))

for i, beta in enumerate(betas):
    # the total cost function that has __call__(x) and gradient(x) with mismatched data op.
    cost_str_mismatched = TotalCost(
        noisy_data, data_operator_mismatched, data_norm,
        structural_prior_operator, prior_norm,
        beta / structural_prior_operator.num_nearest)

    cost_non_str_mismatched = TotalCost(
        noisy_data, data_operator_mismatched, data_norm,
        non_structural_prior_operator, prior_norm,
        beta / non_structural_prior_operator.num_nearest)

    # the total cost function that has __call__(x) and gradient(x) with matched data op.
    cost_str_matched = TotalCost(noisy_data, data_operator, data_norm,
                                 structural_prior_operator, prior_norm,
                                 beta / structural_prior_operator.num_nearest)

    cost_non_str_matched = TotalCost(
        noisy_data, data_operator, data_norm, non_structural_prior_operator,
        prior_norm, beta / non_structural_prior_operator.num_nearest)

    # initial recon
    x0 = np.zeros(data_operator_mismatched.x_shape).ravel()

    # recon using conjugate gradient optimizer, structural prior and mismatched kernel
    res_str_mm = fmin_cg(cost_str_mismatched,
                         x0=x0,
                         fprime=cost_str_mismatched.gradient,
                         maxiter=num_iter,
                         full_output=True)
    recons_str_mismatched[i, :] = res_str_mm[0].reshape(
        data_operator_mismatched.x_shape)

    # recon using conjugate gradient optimizer, structural prior and matched kernel
    res_str_m = fmin_cg(cost_str_matched,
                        x0=x0,
                        fprime=cost_str_matched.gradient,
                        maxiter=num_iter,
                        full_output=True)
    recons_str_matched[i, :] = res_str_m[0].reshape(data_operator.x_shape)

    # recon using conjugate gradient optimizer, non structural prior and mismatched kernel
    res_non_str_mm = fmin_cg(cost_non_str_mismatched,
                             x0=x0,
                             fprime=cost_non_str_mismatched.gradient,
                             maxiter=num_iter,
                             full_output=True)

    recons_non_str_mismatched[i, :] = res_non_str_mm[0].reshape(
        data_operator_mismatched.x_shape)

    # recon using conjugate gradient optimizer, non structural prior and matched kernel
    res_non_str_m = fmin_cg(cost_non_str_matched,
                            x0=x0,
                            fprime=cost_non_str_matched.gradient,
                            maxiter=num_iter,
                            full_output=True)

    recons_non_str_matched[i, :] = res_non_str_m[0].reshape(
        data_operator.x_shape)

#--------------------------------------------------------------------------------
# show the results
fig, ax = plt.subplots(2,
                       betas.size,
                       figsize=(betas.size * 3, 2 * 3),
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
                  recons_non_str_mismatched[i, :],
                  color='tab:orange',
                  label='reconstruction')

    ax[1, i].fill_between(np.arange(n),
                          x_true,
                          0 * x_true + x_true.min(),
                          color=(0.8, 0.8, 0.8))
    ax[1, i].plot(np.arange(n),
                  noisy_data,
                  color='tab:blue',
                  label='blurry and noisy signal')
    ax[1, i].plot(np.arange(n),
                  recons_str_mismatched[i, :],
                  color='tab:red',
                  label='it. recon')

    ax[0, i].set_title(f'prior weight {beta}', fontsize='medium')

ax[0, -1].legend(fontsize='small')
ax[1, -1].legend(fontsize='small')
ax[0, 0].set_ylabel('non structural prior')
ax[1, 0].set_ylabel('structural prior')

for axx in ax.ravel():
    axx.set_ylim(-0.2, 1.3 * x_true.max())
    axx.grid(ls=':')

fig.suptitle('mismatched kernel')
fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots(2,
                         betas.size,
                         figsize=(betas.size * 3, 2 * 3),
                         sharex=True,
                         sharey=True)
for i, beta in enumerate(betas):
    ax2[0, i].fill_between(np.arange(n),
                           x_true,
                           0 * x_true + x_true.min(),
                           color=(0.8, 0.8, 0.8))
    ax2[0, i].plot(np.arange(n),
                   noisy_data,
                   color='tab:blue',
                   label='blurry and noisy signal')
    ax2[0, i].plot(np.arange(n),
                   recons_non_str_matched[i, :],
                   color='tab:orange',
                   label='it. recon')

    ax2[1, i].fill_between(np.arange(n),
                           x_true,
                           0 * x_true + x_true.min(),
                           color=(0.8, 0.8, 0.8))
    ax2[1, i].plot(np.arange(n),
                   noisy_data,
                   color='tab:blue',
                   label='blurry and noisy signal')
    ax2[1, i].plot(np.arange(n),
                   recons_str_matched[i, :],
                   color='tab:red',
                   label='reconstruction')

    ax2[0, i].set_title(f'prior weight {beta}', fontsize='medium')

ax2[0, -1].legend(fontsize='small')
ax2[1, -1].legend(fontsize='small')
ax2[0, 0].set_ylabel('non structural prior')
ax2[1, 0].set_ylabel('structural prior')

for axx in ax2.ravel():
    axx.set_ylim(-0.2, 1.3 * x_true.max())
    axx.grid(ls=':')

fig2.suptitle('matched kernel')
fig2.tight_layout()
fig2.show()

fig3, ax3 = plt.subplots(1, 2, figsize=(6, 3))
ax3[0].fill_between(np.arange(n),
                    x_true,
                    0 * x_true + x_true.min(),
                    color=(0.8, 0.8, 0.8),
                    label='true signal')
ax3[0].plot(prior_signal, 'k', label='structural signal', lw=0.5)
ax3[0].legend(fontsize='small')
ax3[0].grid(ls=':')
ax3[1].plot(kernel, '.-', label='matched kernel')
ax3[1].plot(kernel_mismatched, '.-', label='mismatched kernel')
ax3[1].legend(fontsize='small')
ax3[1].grid(ls=':')

fig3.tight_layout()
fig3.show()
