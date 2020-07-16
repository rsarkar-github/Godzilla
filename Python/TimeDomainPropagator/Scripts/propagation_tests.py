from ..Propagator.BornScattering import*
import numpy as np
import time


# Set parameters
nz = 400
nx = 400
nt_ = 1000

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

start = time.time()
# Propagate waves
acoustic_propagator(
    vel2d=vel2d_,
    dx=dx_, dz=dz_, dt=dt_, fmax=fmax_,
    source_wavefield=source,
    propagated_wavefield=born_wavefield,
    ncells_pad_z=pad_cells_z,
    ncells_pad_x=pad_cells_x,
    check_params=False
)
end = time.time()
print("Elapsed = %s" % (end - start))
