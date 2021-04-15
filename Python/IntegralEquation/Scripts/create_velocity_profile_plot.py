import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


n = 125
vel_salt = 4.5
vel_water = 1.5
a = 4
b = 9
xmin = -2.5
xmax = 2.5
hz = (b - a) / (n - 1)
hx = (xmax - xmin) / (n - 1)
alpha = 0.5

# Create linearly varying background
vel = np.zeros(shape=(n, n), dtype=np.float32)
for i in range(n):
    vel[i, :] = alpha * (a + i * hz)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

xticks = np.arange(0, n+1, int(n / 3))
xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
yticks = np.arange(0, n+1, int(n / 3))
yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

# Create Gaussian perturbation
pert_gaussian = np.zeros(shape=(n, n), dtype=np.float32)
pert_gaussian[int((n - 1) / 2), int((n - 1) / 2)] = 700.0
pert_gaussian = gaussian_filter(pert_gaussian, sigma=10)
print("Perturbation statistics: max val = ", np.max(pert_gaussian), ", min val = ", np.min(pert_gaussian))
print("Total velocity statistics: max val = ", np.max(vel + pert_gaussian), ", min val = ", np.min(vel + pert_gaussian))

ax = axes[0]
ax.imshow(vel + pert_gaussian, cmap="jet", vmin=vel_water, vmax=vel_salt)
ax.set_xlabel(r"$x_1$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_ylabel(r"$z - 4$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(a)", fontname="Times New Roman", fontsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

print("\n\n")

# Create Salt perturbation
vel_sigsbee = np.load("G:/Research/Freq-Domain/Godzilla/Python/Helmholtz/Data/sigsbee.npz")["arr_0"].T
vel_sigsbee *= 0.3048 * 0.001
vel_sigsbee = np.roll(vel_sigsbee[::2, ::2], shift=30, axis=0)
mask = np.clip(vel_sigsbee, 4.49, 4.5) - 4.49
mask = mask / np.max(mask)
pert_salt = (vel_sigsbee[75:75+n, 150:150+n] - vel) * mask[75:75+n, 150:150+n]
pert_salt = gaussian_filter(pert_salt, sigma=0.75)
print("Perturbation statistics: max val = ", np.max(pert_salt), ", min val = ", np.min(pert_salt))
print("Total velocity statistics: max val = ", np.max(vel + pert_salt), ", min val = ", np.min(vel + pert_salt))

ax = axes[1]
ax.imshow(vel + pert_salt, cmap="jet", vmin=vel_water, vmax=vel_salt)
ax.set_xlabel(r"$x_1$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(b)", fontname="Times New Roman", fontsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks([])
ax.set_yticklabels([])

print("\n\n")

# Create Random perturbation
np.random.seed(seed=5)
pert_random = 3 * np.random.uniform(low=-1.0, high=1.0, size=(n, n))
pert_random = gaussian_filter(pert_random, sigma=5)
print("Perturbation statistics: max val = ", np.max(pert_random), ", min val = ", np.min(pert_random))
print("Total velocity statistics: max val = ", np.max(vel + pert_random), ", min val = ", np.min(vel + pert_random))

ax = axes[2]
im = ax.imshow(vel + pert_random, cmap="jet", vmin=vel_water, vmax=vel_salt)
ax.set_xlabel(r"$x_1$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(c)", fontname="Times New Roman", fontsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks([])
ax.set_yticklabels([])

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, hspace=0.05)
cb_ax = fig.add_axes([0.81, 0.266, 0.01, 0.47])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_label("[km/s]", rotation=270, fontname="Times New Roman", fontsize=10)
cbar_yticks = cbar.get_ticks()
cbar_yticks = cbar_yticks[::2]
cbar.set_ticks(cbar_yticks)
cbar.set_ticklabels(cbar_yticks)

plt.show()
fig.savefig("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Fig/vels.pdf", bbox_inches='tight', pad_inches=0)
