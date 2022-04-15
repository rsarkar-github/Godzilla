import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, gmres
import time, os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
source, rhs = create_source(plot=True, fig_filename="source.pdf", scale1=scale_sol)

#************************************************************
# Define linear operator object
def func_matvec(v):
    v = np.reshape(v, newshape=(nz, n))
    u = v * 0
    op.apply_kernel(u=v*psi, output=u, adj=False, add=False)
    return np.reshape(v - (k ** 2) * u, newshape=(nz * n, 1))

A = LinearOperator(shape=(nz * n, nz * n), matvec=func_matvec, dtype=precision)

# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[])

    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print(closure_variables["counter"], residuals)
    return callback

#************************************************************
# Run GMRES for variable iterations, and also to convergence
def run_gmres(num_iter_list):
    """
    :param num_iter_list: list of positive integers (list of number of GMRES iterations)
    :return: sol_list (list of solutions), actual_sol (solution at convergence)
    """

    sol_list = [np.zeros((nz, n), dtype=precision) for _ in num_iter_list]
    actual_sol = np.zeros((nz, n), dtype=precision)

    print("\n************************************************************")
    print("\nRunning GMRES till convergence...")

    start_t = time.time()
    x, exitcode = gmres(
        A,
        np.reshape(rhs, newshape=(nz * n, 1)),
        maxiter=10000,
        restart=1000,
        callback=make_callback()
    )
    x = np.reshape(x, newshape=(nz, n))
    print("\nConverged with exitcode ", exitcode)
    end_t = time.time()
    print("\nTotal time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

    actual_sol += x

    print("\n************************************************************")
    print("\nRunning GMRES for variable iterations...\n\n")

    for count, num_iter in enumerate(num_iter_list):
        print("\n-------------------------------------------------------------")
        print("\nRunning GMRES for ", num_iter, " iterations...\n")
        start_t = time.time()
        x = gmres(
            A,
            np.reshape(rhs, newshape=(nz * n, 1)),
            maxiter=num_iter,
            restart=num_iter,
            callback=make_callback()
        )[0]
        x = np.reshape(x, newshape=(nz, n))
        end_t = time.time()
        print("\nTotal time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

        sol_list[count] += x

    return sol_list, actual_sol

iter_list = [10, 20, 30]
solutions_list_gmres, solution_true = run_gmres(num_iter_list=iter_list)

def plot_true_sol():

    xticks = np.arange(0, n + 1, int(n / 3))
    xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
    yticks = np.arange(0, nz + 1, int(nz / 3))
    yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

    f = int(np.floor(np.log10(scale_sol)))

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.real(solution_true), cmap="Greys", vmin=-scale_sol, vmax=scale_sol)
    ax.grid(True)
    ax.set_title("True solution (real part)", fontname="Times New Roman", fontsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=10)
    ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)
    ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)

    cbar = plt.colorbar(im)
    cbar_yticks = np.linspace(-scale_sol, scale_sol, 5, endpoint=True)
    if f != 0:
        cbar.ax.text(
            0,
            1.05 * scale_sol,
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
    plt.show()
    fig.savefig(os.path.abspath(_basedir_fig + "/" + "true_solution.pdf"), bbox_inches='tight', pad_inches=0)

plot_true_sol()
#************************************************************
# Sum Born Neumann scattering series for variable iterations
def sum_born_neumann(num_iter_list):
    """
    :param num_iter_list: list of positive integers (list of number of terms to sum)
    :return: sum_list (list of sum of terms of Born Neumann series)
    """
    print("\n************************************************************")
    print("\nSumming Born Neumann series...")

    successive_iter_list = [num_iter_list[0] - 1]
    for kk in range(len(num_iter_list)):
        if kk > 0:
            successive_iter_list.append(num_iter_list[kk] - num_iter_list[kk-1])

    sum_list = [np.zeros((nz, n), dtype=precision) for _ in num_iter_list]
    norm_list = [np.linalg.norm(rhs)]

    series_sum = rhs * 1.0
    curr_term = rhs * 1.0
    next_term = rhs * 1.0

    for count, kk in enumerate(successive_iter_list):

        print("\n")
        msg = "Summing the next " + str(kk) + " terms"

        for _ in tqdm(range(kk), desc=msg, ncols=100):

            curr_term *= psi
            op.apply_kernel(u=curr_term, output=next_term)
            curr_term *= 0
            curr_term += next_term * (k ** 2)
            series_sum += curr_term
            norm_list.append(np.linalg.norm(series_sum))

        sum_list[count] += series_sum

    return sum_list, norm_list

sum_list_born_neumann, norm_list_born_neumann = sum_born_neumann(num_iter_list=iter_list)

#************************************************************
# Plot comparison, Born-Neumann norm

#------------------------------------------------------------
def make_plot():

    fig = plt.figure()
    plt.semilogy(norm_list_born_neumann, 'ro-', linewidth=2, markersize=6)
    plt.xlabel("Number of terms", fontname="Times New Roman", fontsize=10)
    plt.ylabel("2-Norm of Born-Neumann series sum", fontname="Times New Roman", fontsize=10)
    plt.grid(True)
    plt.show()
    fig.savefig(os.path.abspath(_basedir_fig + "/" + "born_neumann_norm.pdf"), bbox_inches='tight', pad_inches=0)

    #------------------------------------------------------------
    xticks = np.arange(0, n + 1, int(n / 3))
    xticklabels = ["{:4.1f}".format(item) for item in (xmin + xticks * hx)]
    yticks = np.arange(0, nz + 1, int(nz / 3))
    yticklabels = ["{:4.1f}".format(item) for item in (yticks * hz)]

    f = int(np.floor(np.log10(scale_sol)))

    fig, axs = plt.subplots(2, len(iter_list), sharey=True, sharex=True, figsize=(30, 10))

    # Plot GMRES solutions
    for kk in range(len(iter_list)):
        ax = axs[0, kk]
        im = ax.imshow(np.real(solutions_list_gmres[kk]), cmap="Greys", vmin=-scale_sol, vmax=scale_sol)
        ax.grid(True)
        ax.set_title("GMRES, Iteration = " + str(iter_list[kk]), fontname="Times New Roman", fontsize=10)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=10)
        if kk == 0:
            ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar_yticks = np.linspace(-scale_sol, scale_sol, 5, endpoint=True)
        if f != 0:
            cbar.ax.text(
                0,
                1.05 * scale_sol,
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

    # Plot Born-Neumann series sum
    for kk in range(len(iter_list)):

        scale1 = np.max(np.abs(np.real(sum_list_born_neumann[kk])))
        f1 = int(np.floor(np.log10(scale1)))

        ax = axs[1, kk]
        im = ax.imshow(np.real(sum_list_born_neumann[kk]), cmap="Greys", vmin=-scale1, vmax=scale1)
        ax.grid(True)
        ax.set_title("Born-Neumann series, Terms = " + str(iter_list[kk]), fontname="Times New Roman", fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=10)
        ax.set_xlabel(r"$x$  [km]", fontname="Times New Roman", fontsize=10)
        if kk == 0:
            ax.set_ylabel(r"$z$  [km]", fontname="Times New Roman", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar_yticks = np.linspace(-scale1, scale1, 5, endpoint=True)
        cbar.set_ticks(cbar_yticks)
        cbar.set_ticklabels(
            ["{:4.1f}".format(item / (10 ** f1)) for item in cbar_yticks],
            fontname="Times New Roman",
            fontsize=10
        )
        if f1 != 0:
            cbar.ax.text(
                0,
                1.05 * scale1,
                r"$\times$ 1e" + str(f1),
                fontname="Times New Roman",
                fontsize=10
            )

    plt.show()
    fig.savefig(os.path.abspath(_basedir_fig + "/" + "born_neumann_gmres_comp.pdf"), bbox_inches='tight', pad_inches=0)

make_plot()
