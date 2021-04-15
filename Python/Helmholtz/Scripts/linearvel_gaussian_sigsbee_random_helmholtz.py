# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:30:00 2021
@author: rahul
"""


from ..CommonTools.Common import Common
from ..CommonTools.Velocity import Velocity2D
from ..CommonTools.CreateGeometry import CreateGeometry2D
from ..Inversion.CreateMatrixHelmholtz import CreateMatrixHelmholtz2D
from scipy.sparse.linalg import splu
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


n = 125
vel_salt = 4.5
vel_water = 1.5
a1 = 4
b1 = 9
xmin = -2.5
xmax = 2.5
hz = (b1 - a1) / (n - 1)
hx = (xmax - xmin) / (n - 1)
alpha = 0.5

# Define frequency parameters
omega_max = 10 * np.pi
omega_min = 10 * np.pi

# Define grid spacing (in km)
grid_spacing = 0.036

# Create geometry object
cells = n - 1
cells_pad = 30
geom2d = CreateGeometry2D(
    xdim=grid_spacing * cells,
    zdim=grid_spacing * cells,
    vmin=1.8,
    vmax=4.8,
    omega_max=omega_max,
    omega_min=omega_min
)
geom2d.set_params(
    ncells_x=cells,
    ncells_z=cells,
    ncells_x_pad=cells_pad,
    ncells_z_pad=cells_pad,
    check=True
)

# Create linearly varying background
vel_bkg = np.zeros(shape=(n, n), dtype=np.float32)
for i in range(n):
    vel_bkg[i, :] = alpha * (a1 + i * hz)

# Gaussian perturbation
pert_gaussian = np.zeros(shape=(n, n), dtype=np.float32)
pert_gaussian[int((n - 1) / 2), int((n - 1) / 2)] = 700.0
pert_gaussian = gaussian_filter(pert_gaussian, sigma=10)
vel1 = vel_bkg + pert_gaussian

vel = Velocity2D(geometry2d=geom2d)
vel.set_constant_velocity(vel=2.0)
vel.vel[cells_pad: cells_pad+n, cells_pad: cells_pad+n] = vel1

vel.vel[cells_pad: cells_pad+n, 0:cells_pad] = np.reshape(vel1[:, 0], newshape=(n, 1))
vel.vel[cells_pad: cells_pad+n, cells_pad+n:2*cells_pad+n] = np.reshape(vel1[:, n-1], newshape=(n, 1))
vel.vel[0: cells_pad, :] = vel.vel[cells_pad, :]
vel.vel[cells_pad+n:2*cells_pad+n, :] = vel.vel[cells_pad+n-1, :]

mat_helmholtz2d = CreateMatrixHelmholtz2D(velocity2d=vel, pml_damping=Common.pml_damping)
mat = mat_helmholtz2d.create_matrix(omega=10.0 * np.pi)

ngridx = vel.geometry2D.gridpointsX - 2
ngridz = vel.geometry2D.gridpointsZ - 2
b = np.zeros(shape=(ngridx * ngridz), dtype=np.complex64)
b = np.reshape(b, newshape=(ngridx, ngridz))
b[int(ngridx / 2), cells_pad + int(n / 8)] = 1
b = np.reshape(b, newshape=(ngridx * ngridz))

mat1 = splu(mat)
x = mat1.solve(b)
x = np.reshape(x, newshape=(ngridx, ngridz))
x = x[cells_pad-1: cells_pad-1+n, cells_pad-1: cells_pad-1+n]

np.savez("Python/Helmholtz/Data/linearvel_gaussian.npz", x)
x = np.load("Python/Helmholtz/Data/linearvel_gaussian.npz")["arr_0"]

scale = 1e-4
plt.imshow(np.real(x).T, cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real")
plt.show()

# Salt perturbation
vel_sigsbee = np.load("G:/Research/Freq-Domain/Godzilla/Python/Helmholtz/Data/sigsbee.npz")["arr_0"].T
vel_sigsbee *= 0.3048 * 0.001
vel_sigsbee = np.roll(vel_sigsbee[::2, ::2], shift=30, axis=0)
mask = np.clip(vel_sigsbee, 4.49, 4.5) - 4.49
mask = mask / np.max(mask)
pert_salt = (vel_sigsbee[75:75+n, 150:150+n] - vel_bkg) * mask[75:75+n, 150:150+n]
pert_salt = gaussian_filter(pert_salt, sigma=0.75)
vel1 = vel_bkg + pert_salt

vel = Velocity2D(geometry2d=geom2d)
vel.set_constant_velocity(vel=2.0)
vel.vel[cells_pad: cells_pad+n, cells_pad: cells_pad+n] = vel1

vel.vel[cells_pad: cells_pad+n, 0:cells_pad] = np.reshape(vel1[:, 0], newshape=(n, 1))
vel.vel[cells_pad: cells_pad+n, cells_pad+n:2*cells_pad+n] = np.reshape(vel1[:, n-1], newshape=(n, 1))
vel.vel[0: cells_pad, :] = vel.vel[cells_pad, :]
vel.vel[cells_pad+n:2*cells_pad+n, :] = vel.vel[cells_pad+n-1, :]

mat_helmholtz2d = CreateMatrixHelmholtz2D(velocity2d=vel, pml_damping=Common.pml_damping)
mat = mat_helmholtz2d.create_matrix(omega=10.0 * np.pi)

ngridx = vel.geometry2D.gridpointsX - 2
ngridz = vel.geometry2D.gridpointsZ - 2
b = np.zeros(shape=(ngridx * ngridz), dtype=np.complex64)
b = np.reshape(b, newshape=(ngridx, ngridz))
b[cells_pad + int(n / 4), int(ngridz / 2 + 3 * n / 8)] = 1
b = np.reshape(b, newshape=(ngridx * ngridz))

mat1 = splu(mat)
x = mat1.solve(b)
x = np.reshape(x, newshape=(ngridx, ngridz))
x = x[cells_pad-1: cells_pad-1+n, cells_pad-1: cells_pad-1+n]

np.savez("Python/Helmholtz/Data/linearvel_salt.npz", x)
x = np.load("Python/Helmholtz/Data/linearvel_salt.npz")["arr_0"]

scale = 1e-4
plt.imshow(np.real(x).T, cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real")
plt.show()

# Create Random perturbation
np.random.seed(seed=5)
pert_random = 3 * np.random.uniform(low=-1.0, high=1.0, size=(n, n))
pert_random = gaussian_filter(pert_random, sigma=5)

vel1 = vel_bkg + pert_random

vel = Velocity2D(geometry2d=geom2d)
vel.set_constant_velocity(vel=2.0)
vel.vel[cells_pad: cells_pad+n, cells_pad: cells_pad+n] = vel1

vel.vel[cells_pad: cells_pad+n, 0:cells_pad] = np.reshape(vel1[:, 0], newshape=(n, 1))
vel.vel[cells_pad: cells_pad+n, cells_pad+n:2*cells_pad+n] = np.reshape(vel1[:, n-1], newshape=(n, 1))
vel.vel[0: cells_pad, :] = vel.vel[cells_pad, :]
vel.vel[cells_pad+n:2*cells_pad+n, :] = vel.vel[cells_pad+n-1, :]

mat_helmholtz2d = CreateMatrixHelmholtz2D(velocity2d=vel, pml_damping=Common.pml_damping)
mat = mat_helmholtz2d.create_matrix(omega=10.0 * np.pi)

ngridx = vel.geometry2D.gridpointsX - 2
ngridz = vel.geometry2D.gridpointsZ - 2
b = np.zeros(shape=(ngridx * ngridz), dtype=np.complex64)
b = np.reshape(b, newshape=(ngridx, ngridz))
b[int(ngridx / 2 + 3 * n /8), int(ngridz / 2 + 3 * n /8)] = 1
b = np.reshape(b, newshape=(ngridx * ngridz))

mat1 = splu(mat)
x = mat1.solve(b)
x = np.reshape(x, newshape=(ngridx, ngridz))
x = x[cells_pad-1: cells_pad-1+n, cells_pad-1: cells_pad-1+n]

np.savez("Python/Helmholtz/Data/linearvel_random.npz", x)
x = np.load("Python/Helmholtz/Data/linearvel_random.npz")["arr_0"]

scale = 1e-4
plt.imshow(np.real(x).T, cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real")
plt.show()

# # No perturbation
# vel1 = vel_bkg + 0
#
# vel = Velocity2D(geometry2d=geom2d)
# vel.set_constant_velocity(vel=2.0)
# vel.vel[cells_pad: cells_pad+n, cells_pad: cells_pad+n] = vel1
#
# vel.vel[cells_pad: cells_pad+n, 0:cells_pad] = np.reshape(vel1[:, 0], newshape=(n, 1))
# vel.vel[cells_pad: cells_pad+n, cells_pad+n:2*cells_pad+n] = np.reshape(vel1[:, n-1], newshape=(n, 1))
# vel.vel[0: cells_pad, :] = vel.vel[cells_pad, :]
# vel.vel[cells_pad+n:2*cells_pad+n, :] = vel.vel[cells_pad+n-1, :]
#
# mat_helmholtz2d = CreateMatrixHelmholtz2D(velocity2d=vel, pml_damping=Common.pml_damping)
# mat = mat_helmholtz2d.create_matrix(omega=10.0 * np.pi)
#
# ngridx = vel.geometry2D.gridpointsX - 2
# ngridz = vel.geometry2D.gridpointsZ - 2
# b = np.zeros(shape=(ngridx * ngridz), dtype=np.complex64)
# b = np.reshape(b, newshape=(ngridx, ngridz))
# b[int(ngridx / 2), cells_pad] = 1
# b = np.reshape(b, newshape=(ngridx * ngridz))
#
# mat1 = splu(mat)
# x = mat1.solve(b)
# x = np.reshape(x, newshape=(ngridx, ngridz))
# x = x[cells_pad-1: cells_pad-1+n, cells_pad-1: cells_pad-1+n]
#
# np.savez("Python/Helmholtz/Data/linearvel_nopert.npz", x)
# x = np.load("Python/Helmholtz/Data/linearvel_nopert.npz")["arr_0"]
#
# scale = 1e-4
# plt.imshow(np.real(x).T, cmap="Greys", vmin=-scale, vmax=scale)
# plt.grid(True)
# plt.title("Real")
# plt.show()
