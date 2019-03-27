from Common import*
from CreateGeometry import CreateGeometry2D
from Acquisition import Acquisition2D
from Velocity import Velocity2D
from TfwiLeastSquares import TfwiLeastSquares2D
import numpy as np


# Define frequency parameters (in Hertz)
freq_peak_ricker = 18
freq_max = 30
freq_min = 6.67
flat_spectrum = False
gaussian_spectrum = True
omega_max = 2 * Common.pi * freq_max
omega_min = 2 * Common.pi * freq_min
omega_mean = 2 * Common.pi * freq_peak_ricker
omega_std = (omega_max - omega_min) * 0.3
taper_pct = 0.1
dt = 0.5 / freq_max
nt = 100
domega = (2 * Common.pi) / (nt * dt)
delay = 0.1

# Create geometry object
geom2d = CreateGeometry2D(
    xdim=3.0,
    zdim=2.0,
    vmin=1.5,
    vmax=3.0,
    omega_max=omega_max,
    omega_min=omega_min
)
geom2d.set_default_params()

print("Number of grid points in X", geom2d.gridpointsX)
print("Number of grid points in Z", geom2d.gridpointsZ)
print("Number of cells in X", geom2d.ncellsX)
print("Number of cells in Z", geom2d.ncellsZ)
print("Number of pad cells in X", geom2d.ncellsX_pad)
print("Number of pad cells in Z", geom2d.ncellsZ_pad)

# Create acquisition object
skip_src = 10
skip_rcv = 1
acq2d = Acquisition2D(geometry2d=geom2d)
acq2d.set_split_spread_acquisition(source_skip=skip_src, receiver_skip=skip_rcv, max_offset=2.0)

# Create a default Velocity 2D object
vel_true = Velocity2D(geometry2d=geom2d)
vel_start = Velocity2D(geometry2d=geom2d)
ngridpoints_x = geom2d.gridpointsX
ngridpoints_z = geom2d.gridpointsZ

# Put Gaussian perturbation
sigma_x_gaussian = 0.3
sigma_z_gaussian = 0.3
center_nx = int(ngridpoints_x / 2)
center_nz = int(ngridpoints_z / 2.5)

vel_true.set_constant_velocity(vel=3.0)
vel_true.create_gaussian_perturbation(
    dvel=-1.5,
    sigma_x=sigma_x_gaussian,
    sigma_z=sigma_z_gaussian,
    nx=center_nx,
    nz=center_nz
)
vel = vel_true.vel
vel[:, center_nz + 199: center_nz + 200] = 2.0
vel_true.vel = vel

vel_start.set_constant_velocity(vel=3.0)
vel_start.create_gaussian_perturbation(
    dvel=-1.5,
    sigma_x=sigma_x_gaussian,
    sigma_z=sigma_z_gaussian,
    nx=center_nx,
    nz=center_nz
)

# Create a Tfwi object
tfwilsq = TfwiLeastSquares2D(veltrue=vel_true, velstart=vel_start, acquisition=acq2d)

tfwilsq.veltrue.plot(
    title="True Model",
    pad=False,
    vmin=1.5,
    vmax=3.0,
    xlabel="X grid points",
    ylabel="Z grid points",
    savefile="Fig/veltrue-anomaly.pdf"
)
tfwilsq.velstart.plot(
    title="Starting Model",
    pad=False,
    vmin=1.5,
    vmax=3.0,
    xlabel="X grid points",
    ylabel="Z grid points",
    savefile="Fig/velstart-anomaly.pdf"
)
tfwilsq.veltrue.plot_difference(
    vel_other=tfwilsq.velstart,
    pad=False,
    title="Model Difference",
    xlabel="X grid points",
    ylabel="Z grid points",
    vmin=-0.5,
    vmax=0.5,
    cmap="Greys",
    savefile="Fig/veldiff-anomaly.pdf"
)

omega_list = np.arange(omega_min, omega_max, (omega_max - omega_min) / 50.0).tolist()
tfwilsq.omega_list = omega_list

if not flat_spectrum:
    if not gaussian_spectrum:
        tfwilsq.set_ricker_wavelet(omega_peak=2.0 * Common.pi * freq_peak_ricker)
    else:
        tfwilsq.set_gaussian_wavelet(omega_mean=omega_mean, omega_std=omega_std)
else:
    tfwilsq.set_flat_spectrum_wavelet()

tfwilsq.apply_frequency_taper(
    omega_low=omega_min,
    omega_high=omega_max,
    omega1=omega_min + (omega_max - omega_min) * taper_pct,
    omega2=omega_max - (omega_max - omega_min) * taper_pct
)

inverted_model, inversion_metrics = tfwilsq.perform_lsm_cg(
    epsilon=0.2,
    gamma=0,
    niter=30,
    save_lsm_image=True,
    save_lsm_allimages=True,
    lsm_image_file="Fig/lsm-inverted-image-anomaly0.3-maxoff2.0-eps0.2",
    lsm_image_data_file="Data/lsm-inverted-image-anomaly0.3-maxoff2.0-eps0.2",
    save_lsm_adjoint_image=True,
    save_lsm_adjoint_allimages=False,
    lsm_adjoint_image_file="Fig/lsm-adjoint-image-anomaly0.3-maxoff2.0",
    lsm_adjoint_image_data_file="Data/lsm-adjoint-image-anomaly0.3-maxoff2.0"
)

print(inversion_metrics)
