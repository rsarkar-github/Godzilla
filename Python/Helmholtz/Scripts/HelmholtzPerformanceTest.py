# -*- coding: utf-8 -*-
"""
Created on Mon May 07 09:30:00 2018
@author: rahul
"""

from ..CommonTools.Common import Common
from ..CommonTools.Velocity import Velocity2D
from ..CommonTools.CreateGeometry import CreateGeometry2D
from ..Inversion.CreateMatrixHelmholtz import CreateMatrixHelmholtz2D
from scipy.sparse.linalg import splu
import numpy as np
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


# Global parameters
ncells_list = range(50, 150, 50)
nfac = 5
nsolve = 10

# Define frequency parameters
omega_max = 125.0
omega_min = 3.0

# Define grid spacing (in km)
grid_spacing = 0.01

# Create list for storing results
gridpoints_list = []
fac_time_list = []
solve_time_list = []

for cells in ncells_list:

    # Print number of cells
    print("\n\nCells = ", cells, "\n")

    # Create geometry object
    geom2d = CreateGeometry2D(
        xdim=grid_spacing * cells,
        zdim=grid_spacing * cells,
        vmin=1.5,
        vmax=2.5,
        omega_max=omega_max,
        omega_min=omega_min
    )
    geom2d.set_params(
        ncells_x=cells,
        ncells_z=cells,
        ncells_x_pad=75,
        ncells_z_pad=75,
        check=False
    )

    # Get number of grid points
    gridpoints = (geom2d.gridpointsX - 2) * (geom2d.gridpointsZ - 2)
    gridpoints_list.append(gridpoints * 0.000001)

    # Create a default Velocity 2D object
    vel = Velocity2D(geometry2d=geom2d)

    # Create a MatrixHelmholtz2D object
    mat_helmholtz2d = CreateMatrixHelmholtz2D(velocity2d=vel, pml_damping=Common.pml_damping)
    mat = mat_helmholtz2d.create_matrix(omega=50.0)

    # Create rhs
    ngridx = vel.geometry2D.gridpointsX - 2
    ngridz = vel.geometry2D.gridpointsZ - 2
    b = np.zeros(shape=(ngridx * ngridz), dtype=np.complex64)
    b[int(ngridx * ngridz / 2) + 300] = 1.0

    # Factorize matrix nfac times
    print("Starting factorization test...")
    cum_fac_time = 0.0
    for _ in range(nfac):
        fac_time_start = time.time()
        mat1 = splu(mat)
        fac_time_end = time.time()
        fac_time = fac_time_end - fac_time_start
        cum_fac_time += fac_time
    fac_time_list.append(cum_fac_time / float(nfac))

    # Perform a solve
    print("Starting solve test...")
    mat1 = splu(mat)
    cum_solve_time = 0.0
    for _ in range(nsolve):
        solve_time_start = time.time()
        x = mat1.solve(b)
        solve_time_end = time.time()
        solve_time = solve_time_end - solve_time_start
        cum_solve_time += solve_time
    solve_time_list.append(cum_solve_time / float(nsolve))

# Plot figures
# Normal plot
fig = plt.figure()
plt.plot(gridpoints_list, fac_time_list, "-bo", label="$Factorization \; Time$")
plt.plot(gridpoints_list, solve_time_list, "-ro", label="$Solve \; Time \; per \; Shot$")
plt.grid()
plt.legend(loc=2)
ax = fig.axes
ax[0].set_xlabel(
    "Number of grid points in millions $(N)$",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_ylabel(
    "Time in seconds $(T)$",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_title(
    "Time to solve",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
plt.savefig(Common.filepath_base + "Fig/Helmholtz-Performance-Metrics.pdf", bbox_inches="tight")

# Plot figures
# Normal plot 10000 shots
fig = plt.figure()
plt.plot(gridpoints_list, fac_time_list, "-bo", label="$Factorization \; Time$")
plt.plot(gridpoints_list, [x * 10000 for x in solve_time_list], "-ro", label="$Solve \; Time \; for \; 10000\; Shots$")
plt.grid()
plt.legend(loc=2)
ax = fig.axes
ax[0].set_xlabel(
    "Number of grid points in millions $(N)$",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_ylabel(
    "Time in seconds $(T)$",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_title(
    "Time to solve",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
plt.savefig(Common.filepath_base + "Fig/Helmholtz-Performance-Metrics-10000-Shots.pdf", bbox_inches="tight")

# Semilogy plot
fig = plt.figure()
plt.semilogy(gridpoints_list, fac_time_list, "-bo", label="$Factorization \; Time$")
plt.semilogy(gridpoints_list, solve_time_list, "-ro", label="$Solve \; Time \; per \; Shot$")
plt.grid()
plt.legend(loc=2)
ax = fig.axes
ax[0].set_xlabel(
    "Number of grid points in millions $(N)$",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_ylabel(
    "Log of time in seconds $(\log_{10}T$)",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_title(
    "Time to solve",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
plt.savefig(Common.filepath_base + "Fig/Helmholtz-Performance-Metrics-Semilog.pdf", bbox_inches="tight")

# Log-Log plot
fig = plt.figure()
plt.loglog([x * (10 ** 6) for x in gridpoints_list], fac_time_list, "-bo", label="$Factorization \; Time$")
plt.loglog([x * (10 ** 6) for x in gridpoints_list], solve_time_list, "-ro", label="$Solve \; Time \; per \; Shot$")
plt.grid()
plt.legend(loc=2)
ax = fig.axes
ax[0].set_xlabel(
    "Log of number of grid points $(\log_{10}N)$",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_ylabel(
    "Log of time in seconds $(\log_{10}T$)",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
ax[0].set_title(
    "Time to solve",
    fontsize=16,
    fontweight="normal",
    fontname="Times New Roman"
)
plt.savefig(Common.filepath_base + "Fig/Helmholtz-Performance-Metrics-LogLog.pdf", bbox_inches="tight")
