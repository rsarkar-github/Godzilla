# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 09:30:00 2021
@author: rahul
"""

from ..CommonTools.Common import Common
from ..CommonTools.Velocity import Velocity2D
from ..CommonTools.CreateGeometry import CreateGeometry2D
from ..Inversion.CreateMatrixHelmholtz import CreateMatrixHelmholtz2D
from scipy.sparse.linalg import splu
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


n = 101

# Define frequency parameters
omega_max = 18.0
omega_min = 18.0

# Define grid spacing (in km)
grid_spacing = 0.05

# Create geometry object
cells = n - 1
cells_pad = 20
geom2d = CreateGeometry2D(
    xdim=grid_spacing * cells,
    zdim=grid_spacing * cells,
    vmin=1.5,
    vmax=3.0,
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

# Create a default Velocity 2D object
vel = Velocity2D(geometry2d=geom2d)
vel.set_constant_velocity(vel=2.0)
vel1 = np.zeros(shape=(n, n))
vel1[int(n/2), int(n/2)] = 3
vel1 = ndimage.gaussian_filter(100 * vel1, sigma=8)
vel.vel[cells_pad: cells_pad+n, cells_pad: cells_pad+n] += vel1

# Create a MatrixHelmholtz2D object
mat_helmholtz2d = CreateMatrixHelmholtz2D(velocity2d=vel, pml_damping=Common.pml_damping)
mat = mat_helmholtz2d.create_matrix(omega=18.0)

# Create rhs
ngridx = vel.geometry2D.gridpointsX - 2
ngridz = vel.geometry2D.gridpointsZ - 2
b = np.zeros(shape=(ngridx * ngridz), dtype=np.complex64)
b = np.reshape(b, newshape=(ngridx, ngridz))
b[cells_pad-1 + int(n / 4), int(ngridz / 2)] = 1
b = np.reshape(b, newshape=(ngridx * ngridz))

# Solve the system
mat1 = splu(mat)
x = mat1.solve(b)
x = np.reshape(x, newshape=(ngridx, ngridz))
x = x[cells_pad-1: cells_pad-1+n, cells_pad-1: cells_pad-1+n]

np.savez("Python/Helmholtz/Data/gaussian_sol.npz", x)
x = np.load("Python/Helmholtz/Data/gaussian_sol.npz")["arr_0"]

scale = 1e-4
plt.imshow(np.real(x), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real")
plt.colorbar()
plt.show()
