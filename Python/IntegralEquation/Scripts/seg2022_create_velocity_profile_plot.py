import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


n_ = 201
nz_ = 101
a_ = 4.0
b_ = 6.0

vel_salt = 4.5
vel_water = 1.5
xmin = -0.5
xmax = 0.5
hz = (b_ - a_) / (nz_ - 1)
hx = (xmax - xmin) / (n_ - 1)
alpha = 0.5

# Create linearly varying background
vel = np.zeros(shape=(nz_, n_), dtype=np.float32)
for i in range(nz_):
    vel[i, :] = alpha * (a_ + i * hz)

fig, axes = plt.subplots(2, 1, figsize=(8, 16))

xticks = np.arange(0, n_+1, int(n_ / 3))
xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
yticks = np.arange(0, nz_+1, int(nz_ / 3))
yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

ax = axes[0]
ax.imshow(vel, cmap="jet", vmin=vel_water, vmax=vel_salt)
ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(a)", fontname="Times New Roman", fontsize=10)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

print("\n\n")

# Create Salt perturbation
vel_sigsbee = np.load("D:/Research/Freq-Domain/Godzilla/Python/Helmholtz/Data/sigsbee.npz")["arr_0"].T
vel_sigsbee *= 0.3048 * 0.001
vel_sigsbee = np.roll(vel_sigsbee[::2, ::2], shift=30, axis=0)
mask = np.clip(vel_sigsbee, 4.49, 4.5) - 4.49
mask = mask / np.max(mask)
pert_salt = (vel_sigsbee[75:75+nz_, 150:150+n_] - vel) * mask[75:75+nz_, 150:150+n_]
pert_salt = gaussian_filter(pert_salt, sigma=0.5)
print("Perturbation statistics: max val = ", np.max(pert_salt), ", min val = ", np.min(pert_salt))
print("Total velocity statistics: max val = ", np.max(vel + pert_salt), ", min val = ", np.min(vel + pert_salt))

ax = axes[1]
im = ax.imshow(vel + pert_salt, cmap="jet", vmin=vel_water, vmax=vel_salt)
ax.set_xlabel(r"$x_1$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(b)", fontname="Times New Roman", fontsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

print("\n\n")

# fig.subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.8, hspace=0.05)
cb_ax = fig.add_axes([0.91, 0.12, 0.03, 0.75])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_label("[km/s]", rotation=270, fontname="Times New Roman", fontsize=10)
cbar_yticks = cbar.get_ticks()
cbar_yticks = cbar_yticks[::2]
cbar.set_ticks(cbar_yticks)
cbar.set_ticklabels(cbar_yticks)

plt.show()
fig.savefig("D:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Fig/seg2022_vels.pdf",
            bbox_inches='tight', pad_inches=0)
