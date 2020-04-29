from ..Propagator.BornScattering import*
from ...Utilities.LinearSolvers import*
import numpy as np
import matplotlib.pyplot as plt


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
vel_pert2d_ = np.zeros((nt_, nz, nx), dtype=np.float32)
# vel_pert2d_[:, int(nz / 2) - 50, pad_cells_x:(nx - pad_cells_x)] = 1
vel_pert2d_[:, int(nz / 2) - 50, int(nx / 2)] = 1
output = np.zeros((nt_, nz, nx), dtype=np.float32)

# Create source wavefield and resceiver mask
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
# receiver_restriction_mask[pad_cells_z + 1, int(nx / 2) + 100] = 1.0

# Create data to invert (apply adjoint to recorded data)
recorded_data = np.zeros((nt_, nz, nx), dtype=np.float32)
born_time_dependent_pert_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=vel_pert2d_,
    source_wavefield=source,
    born_scattered_wavefield=recorded_data,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    adjoint_mode=False
)
recorded_data *= np.reshape(receiver_restriction_mask, newshape=(1, nz, nx))
born_time_dependent_pert_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=vel_pert2d_,
    source_wavefield=source,
    born_scattered_wavefield=recorded_data,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    adjoint_mode=True
)

def operator(x, y):
    # Apply normal op
    born_time_dependent_pert_normal_op(
        vel2d=vel2d_,
        dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
        vel_pert2d=x,
        output=y,
        source_wavefield=primary_wavefield,
        restriction_mask=receiver_restriction_mask,
        ncells_pad_z=pad_cells_z,
        ncells_pad_x=pad_cells_x,
        check_params=False,
        precomputed_primary_wavefield=True
    )


# Start inversion
niter = 100
inverted_model = np.zeros((nt_, nz, nx), dtype=np.float32)
inverted_model, metrics = conjugate_gradient(
    linear_operator=operator,
    rhs=vel_pert2d_,
    x0=inverted_model,
    niter=niter
)
np.savez("Python/TimeDomainPropagator/Data/inversion1.npz", inverted_model)

# Create data after inversion
modeled_data = np.zeros((nt_, nz, nx), dtype=np.float32)
born_time_dependent_pert_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    vel_pert2d=inverted_model,
    source_wavefield=source,
    born_scattered_wavefield=modeled_data,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    adjoint_mode=False
)
modeled_data *= np.reshape(receiver_restriction_mask, newshape=(1, nz, nx))

recorded_data_plot = recorded_data[:, pad_cells_z + 1, pad_cells_x:(nx - pad_cells_x)]
np.reshape(recorded_data_plot, newshape=(nt_, nx - 2 * pad_cells_x))

modeled_data_plot = modeled_data[:, pad_cells_z + 1, pad_cells_x:(nx - pad_cells_x)]
np.reshape(modeled_data_plot, newshape=(nt_, nx - 2 * pad_cells_x))

plt.subplot(1, 2, 1)
plt.imshow(recorded_data_plot, cmap='Greys')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(modeled_data_plot, cmap='Greys')
plt.colorbar()
plt.show()
