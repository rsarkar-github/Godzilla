# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:30:00 2021
@author: rahul
"""


import numpy as np
import matplotlib.pyplot as plt

n = 125
a1 = 4
b1 = 9
xmin = -2.5
xmax = 2.5
hz = (b1 - a1) / (n - 1)
hx = (xmax - xmin) / (n - 1)

x1 = np.load("G://Research/Freq-Domain/Godzilla/Python/Helmholtz/Data/linearvel_gaussian.npz")["arr_0"]
x1 = np.real(x1).T
x2 = np.load("G://Research/Freq-Domain/Godzilla/Python/Helmholtz/Data/linearvel_salt.npz")["arr_0"]
x2 = np.real(x2).T
x3 = np.load("G://Research/Freq-Domain/Godzilla/Python/Helmholtz/Data/linearvel_random.npz")["arr_0"]
x3 = np.real(x3).T
scale = 1e-4

fig, axes = plt.subplots(3, 1, figsize=(6, 18))
xticks = np.arange(0, n+1, int(n / 3))
xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
yticks = np.arange(0, n+1, int(n / 3))
yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

ax = axes[0]
ax.imshow(x1, cmap="Greys", vmin=-scale, vmax=scale)
ax.set_ylabel(r"$z - 4$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(a)", fontname="Times New Roman", fontsize=10)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.scatter([int(n / 2)], [int(n / 8)], c='r', s=40)

ax = axes[1]
ax.imshow(x2, cmap="Greys", vmin=-scale, vmax=scale)
ax.set_ylabel(r"$z - 4$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(b)", fontname="Times New Roman", fontsize=10)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.scatter([int(n / 4)], [int(n / 2 + 3 * n / 8)], c='r', s=40)

ax = axes[2]
ax.imshow(x3, cmap="Greys", vmin=-scale, vmax=scale)
ax.set_xlabel(r"$x_1$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_ylabel(r"$z - 4$  [km]", fontname="Times New Roman", fontsize=10)
ax.set_title("(c)", fontname="Times New Roman", fontsize=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.scatter([int(n / 2 + 3 * n / 8)], [int(n / 2 + 3 * n / 8)], c='r', s=40)

plt.show()
fig.savefig("G:/Research/Freq-Domain/Godzilla/Python/Helmholtz/Fig/sols.pdf", bbox_inches='tight', pad_inches=0)
