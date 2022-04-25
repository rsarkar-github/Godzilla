import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, gmres
import time, sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..Solver import HelmholtzOperators
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel2d as Lipp2d


n = 201
nz = 201
a = 4
b = 5
xmin = -0.5
xmax = 0.5
hz = (b - a) / (nz - 1)
hx = (xmax - xmin) / (n - 1)
alpha = 0.5

omega = 30 * 2 * np.pi
k = omega / alpha
m = 1000
precision = np.complex128

if len(sys.argv) < 3:
    ValueError("Input parameters to program not provided. Run program as\n"
               "python -m program run_number(str) model_mode(int)")
run_number = sys.argv[1]
model_mode = int(sys.argv[2])

#************************************************************
# Create directories if they don't exist
_file = os.path.basename(__file__)[:-3]
_basedir = "./Python/IntegralEquation/Runs/" + _file + "/" + run_number
if not os.path.exists(os.path.abspath(_basedir)):
    os.makedirs(os.path.abspath(_basedir))

_basedir_data = _basedir + "/Data"
if not os.path.exists(os.path.abspath(_basedir_data)):
    os.makedirs(os.path.abspath(_basedir_data))

_basedir_fig = _basedir + "/Fig"
if not os.path.exists(os.path.abspath(_basedir_fig)):
    os.makedirs(os.path.abspath(_basedir_fig))

#************************************************************
# Create linearly varying background
vel = np.zeros(shape=(nz, n), dtype=np.float64)
for i in range(nz):
    vel[i, :] = alpha * (a + i * hz)

#************************************************************
# Create perturbation fields
def create_pert_fields(mode, plot=False, fig_filename="fig.pdf"):
    """
    :param mode: int
        0 - Gaussian + reflector
        1 - Salt
        2 - Three scatterers
    :param plot: bool (if True then plot, else do not plot)
    :param fig_filename: str (figure file name)
    :return: total_vel_, pert_, psi_
    """
    if mode == 0:
        # Create Gaussian perturbation
        pert1_ = np.zeros(shape=(nz, n), dtype=np.float64)
        pert1_[int((nz - 1) / 2), int((n - 1) / 2)] = 4000.0
        pert1_ = gaussian_filter(pert1_, sigma=20)

        # Create flat reflector
        pert2_ = np.zeros(shape=(nz, n), dtype=np.float64)
        pert2_[int((nz - 1) * 0.75): int((nz - 1) * 0.77), int((n - 1) * 0.1): int((n - 1) * 0.9)] = 3.0
        pert2_ = gaussian_filter(pert2_, sigma=2)

        pert_ = pert1_ + pert2_

    if mode == 1:
        # Create Salt perturbation
        vel_sigsbee = np.load(os.path.abspath("./Python/Helmholtz/Data/sigsbee.npz"))["arr_0"].T
        vel_sigsbee *= 0.3048 * 0.001
        vel_sigsbee = np.roll(vel_sigsbee[::2, ::2], shift=30, axis=0)
        mask = np.clip(vel_sigsbee, 4.49, 4.5) - 4.49
        mask = mask / np.max(mask)
        pert_salt = (vel_sigsbee[75:75 + nz, 150:150 + n] - vel) * mask[75:75 + nz, 150:150 + n]
        pert_ = gaussian_filter(pert_salt, sigma=0.5)

    if mode == 2:
        # Create three scatterers
        pert1_ = np.zeros(shape=(nz, n), dtype=np.float64)
        pert1_[int((nz - 1) / 3), int((n - 1) / 2)] = 1500.0
        pert1_ = gaussian_filter(pert1_, sigma=10)

        pert2_ = np.zeros(shape=(nz, n), dtype=np.float64)
        pert2_[int((nz - 1) * 2 / 3), int((n - 1) / 3)] = 1500.0
        pert2_ = gaussian_filter(pert2_, sigma=10)

        pert3_ = np.zeros(shape=(nz, n), dtype=np.float64)
        pert3_[int((nz - 1) * 2 / 3), int((n - 1) * 2 / 3)] = 1500.0
        pert3_ = gaussian_filter(pert3_, sigma=10)

        pert_ = pert1_ + pert2_ + pert3_

    # Create 2D velocity and perturbation fields (psi)
    total_vel_ = vel + pert_
    psi_ = (alpha ** 2) * (1.0 / (vel ** 2) - 1.0 / (total_vel_ ** 2))
    psi_ = psi_.astype(precision)

    # Plotting
    if plot:

        xticks = np.arange(0, n + 1, int(n / 3))
        xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
        yticks = np.arange(0, nz + 1, int(nz / 3))
        yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(30, 10))

        # Plot velocity
        ax = axs[0]
        im0 = ax.imshow(vel, cmap="jet", vmin=2.0, vmax=4.5)
        ax.grid(True)
        ax.set_title("Background Velocity", fontname="Times New Roman", fontsize=15)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)
        ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im0, cax=cax)
        cbar.set_label("[km/s]", rotation=270, fontname="Times New Roman", fontsize=10, labelpad=10)
        cbar_yticks = cbar.get_ticks()
        cbar_yticks = cbar_yticks[::2]
        cbar.set_ticks(cbar_yticks)
        cbar.set_ticklabels(
            ["{:4.1f}".format(item) for item in cbar_yticks],
            fontname="Times New Roman",
            fontsize=10
        )

        # Plot total velocity
        ax = axs[1]
        im1 = ax.imshow(total_vel_, cmap="jet", vmin=2.0, vmax=4.5)
        ax.grid(True)
        ax.set_title("Total Velocity", fontname="Times New Roman", fontsize=15)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im1, cax=cax)
        cbar.set_label("[km/s]", rotation=270, fontname="Times New Roman", fontsize=10, labelpad=10)
        cbar_yticks = cbar.get_ticks()
        cbar_yticks = cbar_yticks[::2]
        cbar.set_ticks(cbar_yticks)
        cbar.set_ticklabels(
            ["{:4.1f}".format(item) for item in cbar_yticks],
            fontname="Times New Roman",
            fontsize=10
        )

        # Plot perturbation
        scale = np.max(np.abs(pert_))
        ax = axs[2]
        im2 = ax.imshow(pert_, cmap="Greys", vmin=-scale, vmax=scale)
        ax.grid(True)
        ax.set_title("Perturbation", fontname="Times New Roman", fontsize=15)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im2, cax=cax)
        cbar.set_label("[km/s]", rotation=270, fontname="Times New Roman", fontsize=10, labelpad=10)
        cbar_yticks = np.linspace(-scale, scale, 5, endpoint=True)
        cbar.set_ticks(cbar_yticks)
        cbar.set_ticklabels(
            ["{:4.1f}".format(item) for item in cbar_yticks],
            fontname="Times New Roman",
            fontsize=10
        )

        plt.show()

        fig.savefig(os.path.abspath(_basedir_fig + "/" + fig_filename), bbox_inches='tight', pad_inches=0)

    return total_vel_, pert_, psi_

total_vel, pert, psi = create_pert_fields(mode=model_mode, plot=True, fig_filename="vels.pdf")

#************************************************************
# Initialize operator
def init_op(green_func_filepath):
    """
    :param green_func_filepath: path of green's function file
    :return: op_ (Lipp2d operator)
    """

    if os.path.exists(green_func_filepath):
        op_ = Lipp2d(
            n=n,
            nz=nz,
            k=k,
            a=a,
            b=b,
            m=m,
            precision=precision,
            light_mode=True
        )

        temp = np.load(green_func_filepath)["green_func"]
        op_.set_parameters(
            n=n,
            nz=nz,
            k=k,
            a=a,
            b=b,
            m=m,
            precision=precision,
            green_func=temp
        )

        del temp

    else:
        op_ = Lipp2d(
            n=n,
            nz=nz,
            k=k,
            a=a,
            b=b,
            m=m,
            precision=precision,
            light_mode=False
        )
        op_.write_green_func(green_func_file=green_func_filepath)

    return op_

green_func_filename = "green_func.npz"
path = os.path.abspath(_basedir_data + "/" + green_func_filename)
op = init_op(green_func_filepath=path)

#************************************************************
# Create source
def create_source(plot=False, fig_filename="fig.pdf", scale=1.0, scale1=1e-5):
    """
    :param plot: bool (if True then plot, else do not plot)
    :param fig_filename: str (figure file name)
    :param scale: float (positive, for colorbar of f_)
    :param scale1: float (positive, for colorbar of rhs_)
    :return: f_, rhs_
    """
    # Source
    xgrid = np.linspace(start=xmin, stop=xmax, num=n, endpoint=True)
    zgrid = np.linspace(start=a, stop=b, num=nz, endpoint=True)
    p = 0.0
    q = a + (b - a) / 10.0
    sigma = 0.025
    z, x1 = np.meshgrid(zgrid, xgrid / 1, indexing="ij")
    distsq = (z - q) ** 2 + (x1 - p) ** 2
    f_ = np.exp(-0.5 * distsq / (sigma ** 2))
    f_ = f_.astype(precision)

    # Create LSE rhs
    rhs_ = np.zeros((nz, n), dtype=precision)
    start_t = time.time()
    op.apply_kernel(u=f_, output=rhs_)
    end_t = time.time()
    print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")
    print("Finished LSE rhs computation\n")

    if plot:

        xticks = np.arange(0, n + 1, int(n / 3))
        xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
        yticks = np.arange(0, nz + 1, int(nz / 3))
        yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

        f = int(np.floor(np.log10(scale)))
        f1 = int(np.floor(np.log10(scale1)))

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(30, 10))

        # Plot source
        ax = axs[0]
        im0 = ax.imshow(np.real(f_), cmap="Greys", vmin=-scale, vmax=scale)
        ax.grid(True)
        ax.set_title("Source", fontname="Times New Roman", fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)
        ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im0, cax=cax)
        cbar_yticks = np.linspace(-scale, scale, 5, endpoint=True)
        if f != 0:
            cbar.ax.text(
                0,
                1.05 * scale,
                r"$\times$ 1e" + str(f),
                fontname="Times New Roman",
                fontsize=10
            )
        cbar.set_ticks(cbar_yticks)
        cbar.set_ticklabels(
            ["{:4.1f}".format(item / (10 ** f)) for item in cbar_yticks],
            fontname="Times New Roman",
            fontsize=10
        )

        # Plot LSE rhs (real)
        ax = axs[1]
        im1 = ax.imshow(np.real(rhs_), cmap="Greys", vmin=-scale1, vmax=scale1)
        ax.grid(True)
        ax.set_title("Lippmann-Schwinger RHS (real part)", fontname="Times New Roman", fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im1, cax=cax)
        cbar_yticks = np.linspace(-scale1, scale1, 5, endpoint=True)
        if f1 != 0:
            cbar.ax.text(
                0,
                1.05 * scale1,
                r"$\times$ 1e" + str(f1),
                fontname="Times New Roman",
                fontsize=10
            )
        cbar.set_ticks(cbar_yticks)
        cbar.set_ticklabels(
            ["{:4.1f}".format(item / (10 ** f1)) for item in cbar_yticks],
            fontname="Times New Roman",
            fontsize=10
        )

        # Plot LSE rhs (imag)
        ax = axs[2]
        im2 = ax.imshow(np.imag(rhs_), cmap="Greys", vmin=-scale1, vmax=scale1)
        ax.grid(True)
        ax.set_title("Lippmann-Schwinger RHS (imag part)", fontname="Times New Roman", fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im2, cax=cax)
        cbar_yticks = cbar.get_ticks()
        cbar_yticks = cbar_yticks[::2]
        if f1 != 0:
            cbar.ax.text(
                0,
                1.05 * scale1,
                r"$\times$ 1e" + str(f1),
                fontname="Times New Roman",
                fontsize=10
            )
        cbar.set_ticks(cbar_yticks)
        cbar.set_ticklabels(
            ["{:4.1f}".format(item / (10 ** f1)) for item in cbar_yticks],
            fontname="Times New Roman",
            fontsize=10
        )

        plt.show()

        fig.savefig(os.path.abspath(_basedir_fig + "/" + fig_filename), bbox_inches='tight', pad_inches=0)

    return f_, rhs_

scale_sol = 1e-5
f, rhs = create_source(plot=True, fig_filename="source.pdf", scale1=scale_sol)

#************************************************************
# Create Helmholtz matrix
mat = HelmholtzOperators.create_helmholtz2d_matrix(
    a1=xmax - xmin,
    a2=b-a,
    pad1=10,
    pad2=10,
    omega=omega,
    precision=precision,
    vel=total_vel,
    pml_damping=50.0,
    adj=False,
    warnings=True
)

#************************************************************
# Define linear operator objects
def func_matvec(v):
    v = np.reshape(v, newshape=(nz, n))
    u = v * 0
    op.apply_kernel(u=v*psi, output=u, adj=False, add=False)
    return np.reshape(v - (k ** 2) * u, newshape=(nz * n, 1))

linop_lse = LinearOperator(shape=(nz * n, nz * n), matvec=func_matvec, dtype=precision)

def func_matvec1(v):
    v = mat.dot(v)
    v = np.reshape(v, newshape=(nz, n))
    u = v * 0
    op.apply_kernel(u=v, output=u, adj=False, add=False)
    return np.reshape(u, newshape=(nz * n, 1))

linop_lse_left_precond = LinearOperator(shape=(nz * n, nz * n), matvec=func_matvec1, dtype=precision)

def func_matvec2(v):
    v = np.reshape(v, newshape=(nz, n))
    u = v * 0
    op.apply_kernel(u=v, output=u, adj=False, add=False)
    u = np.reshape(u, newshape=(nz * n, 1))
    u = mat.dot(u)
    return u

linop_lse_right_precond = LinearOperator(shape=(nz * n, nz * n), matvec=func_matvec2, dtype=precision)

#************************************************************
# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[])

    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print(closure_variables["counter"], residuals)
    return callback

#************************************************************
# Run GMRES for 4 cases
print("\n************************************************************")
print("\nRunning GMRES for LSE...\n\n")
start_t = time.time()
x1, exitcode = gmres(
    linop_lse,
    np.reshape(rhs, newshape=(nz * n, 1)),
    maxiter=2000,
    restart=2000,
    callback=make_callback()
)
print(exitcode)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

print("\n************************************************************")
print("\nRunning GMRES for Helmholtz...\n\n")
start_t = time.time()
x2, exitcode = gmres(
    mat,
    np.reshape(f, newshape=(nz * n, 1)),
    maxiter=2000,
    restart=2000,
    atol=0,
    tol=1e-3,
    callback=make_callback()
)
print(exitcode)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

print("\n************************************************************")
print("\nRunning GMRES for Helmholtz (left preconditioner)...\n\n")
start_t = time.time()
x3, exitcode = gmres(
    linop_lse_left_precond,
    np.reshape(rhs, newshape=(nz * n, 1)),
    maxiter=2000,
    restart=2000,
    atol=0,
    tol=1e-3,
    callback=make_callback()
)
print(exitcode)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

print("\n************************************************************")
print("\nRunning GMRES for Helmholtz (right preconditioner)...\n\n")
start_t = time.time()
x_, exitcode = gmres(
    linop_lse_right_precond,
    np.reshape(f, newshape=(nz * n, 1)),
    maxiter=2000,
    restart=2000,
    atol=0,
    tol=1e-3,
    callback=make_callback()
)
print(exitcode)
x_ = np.reshape(x_, newshape=(nz, n))
x4 = x_ * 0
op.apply_kernel(u=x_, output=x4)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

#************************************************************
# Plot results
scale = 1e-5
plt.figure()
plt.imshow(np.real(x1), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution): GMRES")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.real(x2), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution): Helmholtz")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.real(x3), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution): Helmholtz (left precond)")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.real(x4), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution): Helmholtz (right precond)")
plt.colorbar()
plt.show()
