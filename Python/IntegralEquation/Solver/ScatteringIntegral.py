import numpy as np
import matplotlib.pyplot as plt


# Define float32 vs float64
type_t = np.float32

# Number of points in the unit interval
N = 1000

# This is the number of points tuple for the cube [-2, 2] x [-2, 2]
grid_points = (int(4 * N), int(4 * N))

# Define some vector u with a spike in the middle
u = np.zeros(shape=grid_points, dtype=type_t)
u[int(grid_points[0] / 2), int(grid_points[1] / 2)] = 1.0

# Compute the action of the operator on u
ufft = np.fft.fft2(u)

# Initialize results and Green's function arrays
x = np.zeros(shape=grid_points, dtype=type_t)
greens_func = np.zeros(shape=grid_points, dtype=type_t)

# Populate the Green's function arrays

