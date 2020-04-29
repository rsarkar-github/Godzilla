import matplotlib.pyplot as plt
import numpy as np


filename = "Python/TimeDomainPropagator/Data/inversion1.npz"
array3d = np.load(filename)["arr_0"]
nt = array3d.shape[0]
nz = array3d.shape[1]
nx = array3d.shape[2]

# Plot parameters
skip = 10
vmin = -1e-1
vmax = 1e-1
dt = 0.004

pad_cells_x = 100
pad_cells_z = 100

# Show movie
for ii in range(0, nt, skip):
    plt.imshow(
        array3d[ii, pad_cells_x:(nx - pad_cells_x), pad_cells_z:(nz - pad_cells_z)],
        cmap='Greys', vmin=vmin, vmax=vmax
    )
    plt.title("T = " + str(ii * dt) + " s")
    plt.colorbar()
    plt.axes().set_aspect("equal")
    plt.pause(0.05)
    plt.gcf().clear()
plt.close()

# Plot stacked image
image_stk = np.sum(array3d, axis=0)[pad_cells_x:(nx - pad_cells_x), pad_cells_z:(nz - pad_cells_z)]
plt.imshow(image_stk, cmap='Greys')
plt.title("Stack")
plt.show()
