from Tfwi import Tfwi2D
from Velocity import*


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
skip_src = 1
skip_rcv = 1
geom2d.set_default_sources(skip=skip_src)
geom2d.set_default_receivers(skip=skip_rcv)

# Create a default Velocity 2D object
vel_true = Velocity2D(geometry2d=geom2d)
ngridpoints_x = geom2d.gridpointsX
ngridpoints_z = geom2d.gridpointsZ

# Put Gaussian perturbation in the center
center_nx = int(ngridpoints_x / 2)
center_nz = int(ngridpoints_z / 2)
vel_true.create_gaussian_perturbation(dvel=0.3, sigma_x=0.03, sigma_z=0.03, nx=center_nx, nz=center_nz)

# Create a Tfwi object, with a constant starting model
tfwi = Tfwi2D(veltrue=vel_true)
tfwi.set_constant_starting_model()
tfwi.veltrue.plot_nopad(
    title="True Model",
    vmin=2.0,
    vmax=2.3,
    xlabel="X grid points",
    ylabel="Z grid points",
    savefile="veltrue.pdf"
)
tfwi.velstart.plot_nopad(
    title="Starting Model",
    vmin=2.0,
    vmax=2.3,
    xlabel="X grid points",
    ylabel="Z grid points",
    savefile="velstart.pdf"
)

omega_list = np.arange(omega_max / 8, omega_max + omega_max / 8, omega_max / 8)
tfwi.set_omega_list(omega_list=omega_list)
tfwi.set_ricker_wavelet(omega_peak=2.0 * Common.pi * freq_peak_ricker)

tfwi.perform_lsm_cg(
    save_lsm_adjoint_image=False,
    save_lsm_adjoint_allimages=True,
    lsm_adjoint_image_file="lsm-adjoint-image-8"
)
