from ..Inversion.TfwiLeastSquares import TfwiLeastSquares2D
from ..CommonTools.Acquisition import Acquisition2D
from ..CommonTools.Velocity import*


# Define frequency parameters (in Hertz)
freq_peak_ricker = 10
freq_max = 20
omega_max = 2 * Common.pi * freq_max
dt = 0.5 / freq_max
nt = 100
domega = (2 * Common.pi) / (nt * dt)
delay = 0.1

# Create geometry object
geom2d = CreateGeometry2D(
    xdim=0.5,
    zdim=0.5,
    vmin=1.5,
    vmax=2.5,
    omega_max=omega_max,
    omega_min=domega
)
geom2d.set_params(
    ncells_x=50,
    ncells_z=50,
    ncells_x_pad=75,
    ncells_z_pad=75,
    check=False
)

# Create default acquisition object
skip_src = 1
skip_rcv = 1
acq2d = Acquisition2D(geometry2d=geom2d)
acq2d.set_default_sources_receivers(source_skip=skip_src, receiver_skip=skip_rcv)

# Create a default Velocity 2D object
vel_true = Velocity2D(geometry2d=geom2d)
ngridpoints_x = geom2d.gridpointsX
ngridpoints_z = geom2d.gridpointsZ

# Put Gaussian perturbation in the center
center_nx = int(ngridpoints_x / 2)
center_nz = int(ngridpoints_z / 2)
vel_true.create_gaussian_perturbation(dvel=0.3, sigma_x=0.03, sigma_z=0.03, nx=center_nx, nz=center_nz)

# Create a Tfwi object, with a constant starting model
tfwilsq = TfwiLeastSquares2D(veltrue=vel_true, velstart=vel_true, acquisition=acq2d)
tfwilsq.set_constant_starting_model()
tfwilsq.veltrue.plot_nopad(
    title="True Model",
    vmin=2.0,
    vmax=2.3,
    xlabel="X grid points",
    ylabel="Z grid points",
    savefile=Common.filepath_base + "Fig/veltrue.pdf"
)
tfwilsq.velstart.plot_nopad(
    title="Starting Model",
    vmin=2.0,
    vmax=2.3,
    xlabel="X grid points",
    ylabel="Z grid points",
    savefile=Common.filepath_base + "Fig/velstart.pdf"
)

omega_list = np.arange(omega_max / 8, omega_max + omega_max / 8, omega_max / 8)
tfwilsq.omega_list = omega_list
tfwilsq.set_ricker_wavelet(omega_peak=2.0 * Common.pi * freq_peak_ricker)

tfwilsq.perform_lsm_cg(
    save_lsm_adjoint_image=False,
    save_lsm_adjoint_allimages=True,
    lsm_adjoint_image_file=Common.filepath_base + "lsm-adjoint-image-8"
)
