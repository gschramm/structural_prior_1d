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

from scipy.ndimage import label, labeled_comprehension

#----------------------------------------------------------------------------------------
# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--noise_level', default=0.06, type=float)
parser.add_argument('--prior_noise_level', default=0.01, type=float)
parser.add_argument('--n', default=256, type=int)
parser.add_argument('--obj_seed', default=0, type=int)
parser.add_argument('--noise_seed', default=0, type=int)
parser.add_argument('--max_iter', default=1000, type=int)
parser.add_argument('--betas',
                    default=[0.003, 0.03, 0.3, 3, 30],
                    type=float,
                    nargs='+')
parser.add_argument('--neighbors',
                    default=[-3, -2, -1, 1, 2, 3],
                    type=int,
                    nargs='+')
parser.add_argument('--num_nearest', default=3, type=int)
parser.add_argument('--sigma', default=1., type=float)
parser.add_argument('--sigma_mismatched', default=0.7, type=float)
parser.add_argument('--add_standalone', action='store_true')
parser.add_argument('--add_extra_grad', action='store_true')
args = parser.parse_args()
print(args)

noise_level: float = args.noise_level
prior_noise_level: float = args.prior_noise_level
n: int = args.n
obj_seed: int = args.obj_seed
noise_seed: int = args.noise_seed
num_iter: int = args.max_iter
betas: npt.NDArray = np.array(args.betas)
neighbors: tuple = tuple(args.neighbors)
num_nearest: int = args.num_nearest
sigma: float = args.sigma
sigma_mismatched: float = args.sigma_mismatched
add_standalone = args.add_standalone
add_extra_grad = args.add_extra_grad

data_norm: SmoothNorm = L2NormSquared()
prior_norm: SmoothNorm = L2NormSquared()

# generate a Gaussian kernel, the last argument of np.linspace can be used to change its width
tmp = np.linspace(-3, 3, 23)
kernel: npt.NDArray = np.exp(-(tmp**2) / (2 * sigma**2))
kernel_mismatched: npt.NDArray = np.exp(-(tmp**2) / (2 * sigma_mismatched**2))

#----------------------------------------------------------------------------------------
np.random.seed(obj_seed)

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
prior_signal = (x_true.max() - x_true)  #* np.cos(3 * x_true)
#prior_signal[(n // 2):] *= -1

# find biggest flat regions to add extra edges in prior signal
dx = x_true[1:] - x_true[:-1]
label_arr, nlabs = label(dx == 0)
labels = np.arange(1, nlabs)
tmp = labeled_comprehension(x_true[:-1], label_arr, labels, lambda x: x.size,
                            int, 0)
max_label = labels[np.argmax(tmp)]

if add_extra_grad:
    start = np.where(label_arr == max_label)[0][0]
    # add mismatch in prior signal
    prior_signal[(start + 10):(start + 15)] += 0.2
    prior_signal[(start + 20):(start + 25)] -= 0.2

# add stand alone feature in true signal
if add_standalone:
    tmp2 = tmp.copy()
    tmp2[np.argmax(tmp)] = tmp.min()
    max_label2 = labels[np.argmax(tmp2)]
    start2 = np.where(label_arr == max_label2)[0]
    start2 = start2[start2.shape[0] // 2]
    x_true[(start2 - 5):(start2 + 5)] += 0.5

# setup the data model
data_operator = Convolution1D(n, kernel)

# generate noise-free data
noise_free_data = data_operator.forward(x_true)

np.random.seed(noise_seed)
noisy_data = noise_free_data + noise_level * x_true.max() * np.random.randn(
    *data_operator.x_shape)
prior_signal += prior_noise_level * prior_signal.max() * np.random.randn(
    *prior_signal.shape)

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
                  label='reconstruction')

    ax[0, i].set_title(f'prior weight {beta}', fontsize='medium')

ax[0, -1].legend(fontsize='small')
ax[1, -1].legend(fontsize='small')
ax[0, 0].set_ylabel('non structural prior')
ax[1, 0].set_ylabel('structural prior')

for axx in ax.ravel():
    axx.set_ylim(-0.2 * x_true.max(), 1.4 * x_true.max())
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
                   label='reconconstruction')

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
    axx.set_ylim(-0.2 * x_true.max(), 1.4 * x_true.max())
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
ax3[0].legend(fontsize='small', loc='lower left')
ax3[0].grid(ls=':')
ax3[0].set_ylim(-0.3, None)

ax3[1].plot(kernel, '.-', label='matched kernel')
ax3[1].plot(kernel_mismatched, '.-', label='mismatched kernel')
ax3[1].legend(fontsize='small')
ax3[1].grid(ls=':')
if add_extra_grad:
    ax3[0].axvline(start + 10, ls='--', color=plt.cm.tab10(2), lw=0.5)
    ax3[0].axvline(start + 25, ls='--', color=plt.cm.tab10(2), lw=0.5)
    for axx in ax[1, :]:
        axx.axvline(start + 10, ls='--', color=plt.cm.tab10(2), lw=0.5)
        axx.axvline(start + 25, ls='--', color=plt.cm.tab10(2), lw=0.5)
    for axx in ax2[1, :]:
        axx.axvline(start + 10, ls='--', color=plt.cm.tab10(2), lw=0.5)
        axx.axvline(start + 25, ls='--', color=plt.cm.tab10(2), lw=0.5)

if add_standalone:
    ax3[0].axvline(start2 - 5, ls='--', color=plt.cm.tab10(4), lw=0.5)
    ax3[0].axvline(start2 + 5, ls='--', color=plt.cm.tab10(4), lw=0.5)
    for axx in ax[1, :]:
        axx.axvline(start2 - 5, ls='--', color=plt.cm.tab10(4), lw=0.5)
        axx.axvline(start2 + 5, ls='--', color=plt.cm.tab10(4), lw=0.5)
    for axx in ax2[1, :]:
        axx.axvline(start2 - 5, ls='--', color=plt.cm.tab10(4), lw=0.5)
        axx.axvline(start2 + 5, ls='--', color=plt.cm.tab10(4), lw=0.5)

fig3.tight_layout()
fig3.show()

fig4, ax4 = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
for i in range(3):
    ax4[i].fill_between(np.arange(n),
                        x_true,
                        0 * x_true + x_true.min(),
                        color=(0.8, 0.8, 0.8),
                        label='true signal')
    ax4[i].grid(ls=':')

ax4[1].plot(noise_free_data)
ax4[2].plot(noisy_data)
ax4[0].set_ylim(ax[0, 0].get_ylim())
fig4.tight_layout()
fig4.show()
