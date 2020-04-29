from ..Propagator.BornScattering import*
import numpy as np
import matplotlib.pyplot as plt


# Set parameters
nz = 400
nx = 500
nt_ = 700

dx_ = 15
dz_ = 15
dt_ = 0.004

fmax_ = 10.0

pad_cells_x = 100
pad_cells_z = 100

# Create velocity, vel pert, output
vel2d_ = np.zeros((nz, nx), dtype=np.float32) + 2000
vel_pert2d_ = np.zeros((nt_, nz, nx), dtype=np.float32)
vel_pert2d_[:, int(nz / 2), pad_cells_x:(nx - pad_cells_x)] = 1
# vel_pert2d_[:, int(nz / 2), int(nx / 2)] = 1
output = np.zeros((nt_, nz, nx), dtype=np.float32)

# Create source wavefield and resceiver mask
source = np.zeros((nt_, nz, nx), dtype=np.float32)
_, vals = ricker_time(freq_peak=fmax_, nt=nt_, dt=dt_, delay=0.15)
vals = vals / np.max(np.abs(vals))
source[:, pad_cells_z + 1, int(nx / 2)] = vals

receiver_restriction_mask = np.zeros((nz, nx), dtype=np.float32)
receiver_restriction_mask[pad_cells_z + 1, pad_cells_x:(nx - pad_cells_x)] = 1.0
# receiver_restriction_mask[pad_cells_z + 1, int(nx / 2) + 100] = 1.0

# Apply normal op
born_time_dependent_pert_normal_op(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=vel_pert2d_,
    output=output,
    source_wavefield=source,
    restriction_mask=receiver_restriction_mask,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x
)

born_image = np.sum(output, axis=0)
plt.imshow(born_image, cmap='Greys')
plt.colorbar()
plt.axes().set_aspect("equal")
plt.show()

# # Show movie
# for ii in range(0, nt_, 20):
#     plt.imshow(vel_pert2d_[ii, :, :], cmap='Greys', vmin=-1e-6, vmax=1e-6)
#     plt.colorbar()
#     plt.axes().set_aspect("equal")
#     plt.pause(0.05)
#     plt.gcf().clear()
