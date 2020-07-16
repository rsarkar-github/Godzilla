from ..Propagator.BornScattering import*
import numpy as np


# Set parameters
nz = 400
nx = 400
nt_ = 500

dx_ = 15
dz_ = 15
dt_ = 0.004

fmax_ = 10.0

pad_cells_x = 100
pad_cells_z = 100

# Create velocity, vel pert, output
vel2d_ = np.zeros((nz, nx), dtype=np.float32) + 2000
vel_pert2d_1 = np.random.uniform(size=(nt_, nz, nx)).astype(np.float32)
vel_pert2d_2 = np.zeros((nt_, nz, nx), dtype=np.float32)
output1 = np.zeros((nt_, nz, nx), dtype=np.float32)
output2 = np.random.uniform(size=(nt_, nz, nx)).astype(np.float32)

# Create source wavefield and receiver mask
source = np.zeros((nt_, nz, nx), dtype=np.float32)
_, vals = ricker_time(freq_peak=fmax_, nt=nt_, dt=dt_, delay=0.15)
vals = vals / np.max(np.abs(vals))
source[:, pad_cells_z + 1, int(nx / 2)] = vals

# Get primary wavefield
primary_wavefield = np.zeros((nt_, nz, nx), dtype=np.float32)
acoustic_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    source_wavefield=source,
    propagated_wavefield=primary_wavefield,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    check_params=False
)

receiver_restriction_mask = np.zeros((nz, nx), dtype=np.float32)
receiver_restriction_mask[pad_cells_z + 1, pad_cells_x:(nx - pad_cells_x)] = 1.0

# Create data to invert (apply adjoint to recorded data)
born_time_dependent_pert_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=vel_pert2d_1,
    source_wavefield=source,
    born_scattered_wavefield=output1,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    adjoint_mode=False
)
output1 *= np.reshape(receiver_restriction_mask, newshape=(1, nz, nx))

output2 *= np.reshape(receiver_restriction_mask, newshape=(1, nz, nx))
born_time_dependent_pert_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=vel_pert2d_2,
    source_wavefield=source,
    born_scattered_wavefield=output2,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    adjoint_mode=True
)

print("Dot product test comparison")
print(np.vdot(output1, output2))
print(np.vdot(vel_pert2d_1, vel_pert2d_2))
