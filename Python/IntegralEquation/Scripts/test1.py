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

fig, ax = plt.subplots(1, 1, figsize=(18, 6))

xticks = np.arange(0, n+1, int(n / 3))
xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
yticks = np.arange(0, n+1, int(n / 3))
yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

# Create Salt perturbation
vel_sigsbee = np.load("D:/Research/Freq-Domain/Godzilla/Python/Helmholtz/Data/sigsbee.npz")["arr_0"].T
vel_sigsbee *= 0.3048 * 0.001
vel_sigsbee = np.roll(vel_sigsbee[::2, ::2], shift=30, axis=0)
mask = np.clip(vel_sigsbee, 4.49, 4.5) - 4.49
mask = mask / np.max(mask)
pert_salt = (vel_sigsbee[75:75+n, 150:150+n] - vel) * mask[75:75+n, 150:150+n]
pert_salt = gaussian_filter(pert_salt, sigma=0.75)
print("Perturbation statistics: max val = ", np.max(pert_salt), ", min val = ", np.min(pert_salt))
print("Total velocity statistics: max val = ", np.max(vel + pert_salt), ", min val = ", np.min(vel + pert_salt))

# im = ax.imshow(vel + pert_salt, cmap="jet", vmin=0, vmax=vel_salt)
im = ax.imshow(vel, cmap="jet", vmin=0, vmax=vel_salt)
# im = ax.imshow(pert_salt, cmap="jet", vmin=0, vmax=vel_salt)
ax.set_xlabel(r"$x_1$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

cb = plt.colorbar(im)
cb.set_label("[km/s]", rotation=270, fontname="Times New Roman", fontsize=10)

plt.show()
fig.savefig("D:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Fig/vels_1.pdf", bbox_inches='tight', pad_inches=0)
