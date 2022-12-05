"""example script for MR reconstruction with smooth cost function using CG and LBFGS"""

import argparse
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from PIL import Image

from operators import BowsherGradient2D, GaussianConv2D
from functionals import SmoothNorm, TotalCost, L2NormSquared

#----------------------------------------------------------------------------------------
# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=200, type=int)
parser.add_argument('--betas',
                    default=[1e-5,1e-4,1e-3,1e-2],
                    type=float,
                    nargs='+')
parser.add_argument('--num_nearest', default=3, type=int)
parser.add_argument('--sigma', default=1.5, type=float)
args = parser.parse_args()
print(args)

num_iter: int = args.max_iter
betas: npt.NDArray = np.array(args.betas)
num_nearest: int = args.num_nearest
sigma: float = args.sigma

data_norm: SmoothNorm = L2NormSquared()
prior_norm: SmoothNorm = L2NormSquared()

xlim = (100,170)
ylim = (70,140)
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- load images
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

x_true = np.array(Image.open('data/t2.png')).astype(np.float64)
x_true /= x_true.max()

prior_image = np.array(Image.open('data/t1.png')).astype(np.float64)
prior_image /= prior_image.max()

data_operator = GaussianConv2D(x_true.shape, sigma)
structural_prior_operator = BowsherGradient2D(x_true.shape, num_nearest = num_nearest,
                                              prior_image = prior_image)
non_structural_prior_operator = BowsherGradient2D(x_true.shape)


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- generate the data
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# generate noise-free data
noise_free_data = data_operator.forward(x_true)

#np.random.seed(noise_seed)
#noisy_data = noise_free_data + noise_level * x_true.max() * np.random.randn(
#    *data_operator.x_shape)
noisy_data = noise_free_data.copy()

#----------------------------------------------------------------------------------------

recons_str_matched = np.zeros((betas.size,) + x_true.shape)
recons_non_str_matched = np.zeros((betas.size,) + x_true.shape)

# initial recon
x0 = np.zeros(data_operator.x_shape).ravel()

for i, beta in enumerate(betas):
    print(f'{i+1}/{len(betas)} beta:{beta}')
    # the total cost function that has __call__(x) and gradient(x) with matched data op.
    cost_str_matched = TotalCost(noisy_data, data_operator, data_norm,
                                 structural_prior_operator, prior_norm,
                                 beta / structural_prior_operator.num_nearest)

    cost_non_str_matched = TotalCost(
        noisy_data, data_operator, data_norm, non_structural_prior_operator,
        prior_norm, beta / non_structural_prior_operator.num_nearest)


    # recon using conjugate gradient optimizer, structural prior and matched kernel
    res_str_m = fmin_cg(cost_str_matched,
                        x0=x0,
                        fprime=cost_str_matched.gradient,
                        maxiter=num_iter,
                        full_output=True)
    recons_str_matched[i, ...] = res_str_m[0].reshape(data_operator.x_shape)

    # recon using conjugate gradient optimizer, non structural prior and matched kernel
    res_non_str_m = fmin_cg(cost_non_str_matched,
                            x0=x0,
                            fprime=cost_non_str_matched.gradient,
                            maxiter=num_iter,
                            full_output=True)

    recons_non_str_matched[i, ...] = res_non_str_m[0].reshape(
        data_operator.x_shape)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# show the results

ims = dict(cmap=plt.cm.Greys_r, vmin = 0, vmax = 1.0)
fs = dict(fontsize='small')

fig, ax = plt.subplots(2,
                       betas.size + 1,
                       figsize=((betas.size + 1) * 3, 2 * 3),
                       sharex=True,
                       sharey=True)
for i, beta in enumerate(betas):
    ax[0, i+1].imshow(recons_non_str_matched[i,...], **ims)
    ax[1, i+1].imshow(recons_str_matched[i,...], **ims)
    ax[0, i+1].set_title(f'deblurred non-structural prior {beta:.1E}', **fs)
    ax[1, i+1].set_title(f'deblurred structural prior {beta:.1E}', **fs)

ax[0,0].imshow(noisy_data, **ims)
ax[0,0].set_title('blurred T2', **fs)
ax[1,0].imshow(x_true, **ims)
ax[1,0].set_title('true T2', **fs)

for axx in ax.ravel():
    axx.set_axis_off()

fig.tight_layout()
fig.show()

# same figure but zoomed
fig2, ax2 = plt.subplots(2,
                       betas.size + 1,
                       figsize=((betas.size + 1) * 3, 2 * 3),
                       sharex=True,
                       sharey=True)
for i, beta in enumerate(betas):
    ax2[0, i+1].imshow(recons_non_str_matched[i,...], **ims)
    ax2[1, i+1].imshow(recons_str_matched[i,...], **ims)
    ax2[0, i+1].set_title(f'deblurred non-structural prior {beta:.1E} (zoom)', **fs)
    ax2[1, i+1].set_title(f'deblurred structural prior {beta:.1E} (zoom)', **fs)

ax2[0,0].imshow(noisy_data, **ims)
ax2[0,0].set_title('blurred T2 (zoom)', **fs)
ax2[1,0].imshow(x_true, **ims)
ax2[1,0].set_title('true T2 (zoom)', **fs)

for axx in ax2.ravel():
    axx.set_axis_off()
    axx.set_xlim(*xlim)
    axx.set_ylim(*ylim)

fig2.tight_layout()
fig2.show()


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#show the structural prior image

fig3, ax3 = plt.subplots(1, 2, figsize=(2*3,3))
ax3[0].imshow(prior_image, **ims)
ax3[0].set_title('structural prior image', **fs)
ax3[1].imshow(prior_image, **ims)
ax3[1].set_title('zoom', **fs)
ax3[1].set_xlim(*xlim)
ax3[1].set_ylim(*ylim)
for axx in ax3.ravel():
    axx.set_axis_off()
fig3.tight_layout()
fig3.show()