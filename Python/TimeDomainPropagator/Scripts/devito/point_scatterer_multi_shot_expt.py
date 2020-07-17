import sys
devito_examples_dir = "/homes/sep/rahul/devito/examples/"
sys.path.append(devito_examples_dir)

from devito import configuration
configuration['log-level'] = 'WARNING'

import DevitoOperators
from DevitoUtils import create_model, conjugate_gradient
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry

import numpy as np
import time

filestr = "point_scatterer_multi_shot"

# Create params dicts
params = {
    "Nx": 300,
    "Nz": 100,
    "Nt": 100,   # this has to be updated later
    "nbl": 75,
    "Ns": 10,
    "Nr": 200,
    "so": 4,
    "to": 2
}

vel = create_model(shape=(params["Nx"], params["Nz"]))
vel.vp.data[:, :] = 2.0

# Simulation time, wavelet
t0 = 0.
tn = 2000.          # Simulation last 2 second (2000 ms)
f0 = 0.010          # Source peak frequency is 10Hz (0.010 kHz)

# Reflection acquisition geometry (sources and receivers are equally spaced in X direction)
src_depth = 20.0                        # Depth is 20m
rec_depth = 20.0                        # Depth is 20m

src_coord = np.empty((params["Ns"], 2))
src_coord[:, 0] = np.linspace(0, vel.domain_size[0], num=params["Ns"])
src_coord[:, 1] = src_depth

rec_coord = np.empty((params["Nr"], 2))
rec_coord[:, 0] = np.linspace(0, vel.domain_size[0], num=params["Nr"])
rec_coord[:, 1] = rec_depth

# Create the geometry objects for background velocity models
src_dummy = np.empty((1, 2))

src_dummy[0, :] = src_coord[int(src_coord.shape[0] / 2), :]
geometry = AcquisitionGeometry(vel, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
params["Nt"] = geometry.nt
del src_dummy

# Define a solver object
solver = AcousticWaveSolver(vel, geometry, space_order=params["so"])

# Create point perturbation
dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
dm[:, int(params["Nx"] / 2), int(params["Nz"] / 2)] = 1.0


# Create wrapper for time dependent Born Hessian
def hessian_wrap(model_pert_in, model_pert_out):
    """
    @Params
    model_pert_in: input numpy array
    model_pert_out: output numpy array
    """
    model_pert_out *= 0.

    DevitoOperators.td_born_hessian(
        model_pert_in=model_pert_in,
        model_pert_out=model_pert_out,
        src_coords=src_coord,
        vel=vel,
        geometry=geometry,
        solver=solver,
        params=params
    )


# Create rhs for inversion
dm_adjoint_image = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
t_start = time.time()
DevitoOperators.td_born_hessian(
    model_pert_in=dm,
    model_pert_out=dm_adjoint_image,
    src_coords=src_coord,
    vel=vel,
    geometry=geometry,
    solver=solver,
    params=params
)
t_end = time.time()
print("\nCreate adjoint image took ", t_end - t_start, " sec")

# Run the inversion
niter = 100
dm_invert, resid = conjugate_gradient(
    hessian_wrap,
    rhs=dm_adjoint_image,
    x0=None,
    niter=niter,
    printobj=False
)

# Save results
np.savez("Data/" + filestr + ".npz", dm_invert, resid)
