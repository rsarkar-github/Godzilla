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

# Create velocity
vel2d_ = np.zeros((nz, nx), dtype=np.float32) + 2000

# Create vel pert
vel_pert2d_ = np.zeros((nt_, nz, nx), dtype=np.float32)
vel_pert2d_[:, int(nz / 2), pad_cells_x:(nx - pad_cells_x)] = 1

# Create source and target wavefields
source = np.zeros((nt_, nz, nx), dtype=np.float32)
born_wavefield = np.zeros((nt_, nz, nx), dtype=np.float32)

_, vals = ricker_time(freq_peak=fmax_, nt=nt_, dt=dt_, delay=0.15)
vals = vals / np.max(np.abs(vals))
source[:, pad_cells_z + 1, int(nx / 2)] = vals

# Forward Born
born_time_dependent_pert_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=vel_pert2d_,
    source_wavefield=source,
    born_scattered_wavefield=born_wavefield,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    adjoint_mode=False
)

# Receiver selection mask
receiver_restriction_mask = np.zeros((nz, nx), dtype=np.float32)
receiver_restriction_mask[pad_cells_z + 1, pad_cells_x:(nx - pad_cells_x)] = 1.0
born_wavefield *= np.reshape(receiver_restriction_mask, newshape=(1, nz, nx))

# recorded_data = born_wavefield[:, pad_cells_z, pad_cells_x:(nx - pad_cells_x)]
# np.reshape(recorded_data, newshape=(nt_, nx - 2 * pad_cells_x))
# plt.imshow(recorded_data, cmap='Greys')
# plt.colorbar()
# plt.axes().set_aspect("equal")
# plt.show()

# Adjoint Born
born_time_dependent_pert_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=vel_pert2d_,
    source_wavefield=source,
    born_scattered_wavefield=born_wavefield,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    adjoint_mode=True
)

# born_image = np.sum(vel_pert2d_, axis=0)
# plt.imshow(born_image, cmap='Greys')
# plt.colorbar()
# plt.axes().set_aspect("equal")
# plt.show()

# Show movie
for ii in range(0, nt_, 20):
    plt.imshow(vel_pert2d_[ii, :, :], cmap='Greys', vmin=-1e-6, vmax=1e-6)
    plt.colorbar()
    plt.axes().set_aspect("equal")
    plt.pause(0.05)
    plt.gcf().clear()
