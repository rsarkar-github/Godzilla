from ..Propagator.BornScattering import*
from ...Utilities.LinearSolvers import*
import numpy as np
import os


# Set parameters
nz = 300
nx = 500
nt_ = 1000

dx_ = 15
dz_ = 15
dt_ = 0.004

fmax_ = 10.0

pad_cells_x = 100
pad_cells_z = 100

# Create velocity, vel pert, output
vel2d_ = np.zeros((nz, nx), dtype=np.float32) + 1500
vel2d_starting = np.zeros((nz, nx), dtype=np.float32) + 2000
vel_pert2d_ = np.zeros((nt_, nz, nx), dtype=np.float32)
vel_pert2d_[:, int(nz / 2), pad_cells_x:(nx - pad_cells_x)] = 1

# Source locations
ns = 1
source_locations = np.zeros((ns, 2))
if ns == 1:
    source_locations[0, 0] = pad_cells_z + 1
    source_locations[0, 1] = int(nx / 2)
else:
    step = (nx - 2 * pad_cells_x) / (ns - 1)
    for j in range(ns):
        source_locations[j, 0] = pad_cells_z + 1
        source_locations[j, 1] = pad_cells_x + 1 + j * step

# Create source wavelet and receiver mask
_, vals = ricker_time(freq_peak=fmax_, nt=nt_, dt=dt_, delay=0.15)
vals = vals / np.max(np.abs(vals))

receiver_restriction_mask = np.zeros((nz, nx), dtype=np.float32)
receiver_restriction_mask[pad_cells_z + 1, pad_cells_x:(nx - pad_cells_x)] = 1.0

def operator(x, y):

    # Zero output
    y *= 0

    # Allocate array for source and primary wavefield
    source = np.zeros((nt_, nz, nx), dtype=np.float32)
    primary_wavefield = np.zeros((nt_, nz, nx), dtype=np.float32)

    # Allocate array for per shot Hessian application
    z = np.zeros((nt_, nz, nx), dtype=np.float32)

    # Loop over shots
    for i in range(ns):

        # Set source wavefield
        source *= 0
        source[:, int(source_locations[i, 0]), int(source_locations[i, 1])] = vals

        # Compute primary wavefield
        acoustic_propagator(
            vel2d=vel2d_starting,
            dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
            source_wavefield=source,
            propagated_wavefield=primary_wavefield,
            ncells_pad_z=pad_cells_z,
            ncells_pad_x=pad_cells_x,
            check_params=False
        )

        # Apply normal op
        born_time_dependent_pert_normal_op(
            vel2d=vel2d_starting,
            dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
            vel_pert2d=x,
            output=z,
            source_wavefield=primary_wavefield,
            restriction_mask=receiver_restriction_mask,
            ncells_pad_z=pad_cells_z,
            ncells_pad_x=pad_cells_x,
            check_params=False,
            precomputed_primary_wavefield=True
        )

        # Add to result
        y += z


def create_rhs(rhs):

    # Allocate array for source and primary wavefield
    source = np.zeros((nt_, nz, nx), dtype=np.float32)

    # Allocate temporary arrays
    image = np.zeros((nt_, nz, nx), dtype=np.float32)
    recorded_data = np.zeros((nt_, nz, nx), dtype=np.float32)

    # Loop over shots
    for i in range(ns):

        # Set source wavefield
        source *= 0
        source[:, int(source_locations[i, 0]), int(source_locations[i, 1])] = vals

        # Forward scattering
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

        # Adjoint application
        born_time_dependent_pert_propagator(
            vel2d=vel2d_starting,
            dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
            vel_pert2d=image,
            source_wavefield=source,
            born_scattered_wavefield=recorded_data,
            ncells_pad_z=pad_cells_z,
            ncells_pad_x=pad_cells_x,
            adjoint_mode=True
        )

        # Add to result
        rhs += image

# Start inversion
niter = 50

if os.path.exists("Python/TimeDomainPropagator/Data/inversion3_single_shot.npz"):
    inverted_model = np.load("Python/TimeDomainPropagator/Data/inversion3_single_shot.npz")["arr_0"]
else:
    inverted_model = np.zeros((nt_, nz, nx), dtype=np.float32)

inversion_rhs = np.zeros((nt_, nz, nx), dtype=np.float32)
create_rhs(inversion_rhs)
inverted_model, metrics = conjugate_gradient(
    linear_operator=operator,
    rhs=inversion_rhs,
    x0=inverted_model,
    niter=niter
)
np.savez("Python/TimeDomainPropagator/Data/inversion3_single_shot.npz", inverted_model, metrics)
