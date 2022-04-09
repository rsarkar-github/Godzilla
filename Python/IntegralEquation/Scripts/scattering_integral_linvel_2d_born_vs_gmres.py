import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, gmres
import time
import os
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

omega = 10 * 2* np.pi
k = omega / alpha
m = 1000
precision = np.complex128

#************************************************************
# Create linearly varying background
vel = np.zeros(shape=(nz, n), dtype=np.float64)
for i in range(nz):
    vel[i, :] = alpha * (a + i * hz)

#************************************************************
# Create perturbation fields
def create_pert_fields(mode, plot=False):
    """
    :param mode:
        0 - Gaussian
        1 - Salt
    :param plot: bool
    :return: total_vel_, pert_, psi_
    """
    if mode == 0:
        # Create Gaussian perturbation
        pert_ = np.zeros(shape=(nz, n), dtype=np.float64)
        pert_[int((nz - 1) / 2), int((n - 1) / 2)] = 4000.0
        pert_ = gaussian_filter(pert_, sigma=20)

    if mode == 1:
        # Create Salt perturbation
        vel_sigsbee = np.load(os.path.abspath("./Python/Helmholtz/Data/sigsbee.npz"))["arr_0"].T
        vel_sigsbee *= 0.3048 * 0.001
        vel_sigsbee = np.roll(vel_sigsbee[::2, ::2], shift=30, axis=0)
        mask = np.clip(vel_sigsbee, 4.49, 4.5) - 4.49
        mask = mask / np.max(mask)
        pert_salt = (vel_sigsbee[75:75 + nz, 150:150 + n] - vel) * mask[75:75 + nz, 150:150 + n]
        pert_ = gaussian_filter(pert_salt, sigma=0.5)

    # Create 2D velocity and perturbation fields (psi)
    total_vel_ = vel + pert_
    psi_ = (alpha ** 2) * (1.0 / (vel ** 2) - 1.0 / (total_vel_ ** 2))
    psi_ = psi_.astype(precision)

    # Plotting
    if plot:

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(30, 10))

        # Plot velocity
        ax = axs[0]
        im0 = ax.imshow(vel, cmap="jet", vmin=2.0, vmax=4.5)
        ax.grid(True)
        ax.set_title("Background Velocity", fontname="Times New Roman", fontsize=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax)

        # Plot total velocity
        ax = axs[1]
        im1 = ax.imshow(total_vel_, cmap="jet", vmin=2.0, vmax=4.5)
        ax.grid(True)
        ax.set_title("Total Velocity", fontname="Times New Roman", fontsize=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)

        # Plot perturbation
        ax = axs[2]
        im2 = ax.imshow(pert_, cmap="Greys", vmin=-1.0, vmax=1.0)
        ax.grid(True)
        ax.set_title("Perturbation", fontname="Times New Roman", fontsize=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)

        plt.show()

    return total_vel_, pert_, psi_

total_vel, pert, psi = create_pert_fields(mode=1, plot=True)

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
path = os.path.abspath("./Python/IntegralEquation/Data/" + green_func_filename)
op = init_op(green_func_filepath=path)

#************************************************************
# Create source
def create_source(plot=False, scale=1.0, scale1=1e-4):

    # Source
    xgrid = np.linspace(start=xmin, stop=xmax, num=n, endpoint=True)
    zgrid = np.linspace(start=a, stop=b, num=nz, endpoint=True)
    p = 0.0
    q = a + (b - a) / 10.0
    sigma = 0.025
    z, x1 = np.meshgrid(zgrid, xgrid / 1, indexing="ij")
    distsq = (z - q) ** 2 + (x1 - p) ** 2
    f = np.exp(-0.5 * distsq / (sigma ** 2))
    f = f.astype(precision)

    # Create LSE rhs
    rhs = np.zeros((nz, n), dtype=precision)
    start_t = time.time()
    op.apply_kernel(u=f, output=rhs)
    end_t = time.time()
    print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")
    print("Finished LSE rhs computation\n")

    if plot:

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(30, 10))

        # Plot source
        ax = axs[0]
        im0 = ax.imshow(np.real(f), cmap="Greys", vmin=-scale, vmax=scale)
        ax.grid(True)
        ax.set_title("Source", fontname="Times New Roman", fontsize=10)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax)

        # Plot LSE rhs
        ax = axs[1]
        im1 = ax.imshow(np.real(rhs), cmap="Greys", vmin=-scale1, vmax=scale1)
        ax.grid(True)
        ax.set_title("Lippmann-Schwinger RHS (real part)", fontname="Times New Roman", fontsize=10)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)

        # Plot LSE rhs
        ax = axs[2]
        im2 = ax.imshow(np.imag(rhs), cmap="Greys", vmin=-scale1, vmax=scale1)
        ax.grid(True)
        ax.set_title("Lippmann-Schwinger RHS (imag part)", fontname="Times New Roman", fontsize=10)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)

        plt.show()

    return f, rhs

source, rhs = create_source(plot=True)

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
# Run GMRES to get total iterations
start_t = time.time()
x, exitCode = gmres(
    A,
    np.reshape(rhs, newshape=(nz * n, 1)),
    maxiter=1000,
    restart=100,
    callback=make_callback()
)
print(exitCode)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")
print("Residual norm = ", np.linalg.norm(rhs - np.reshape(A.matvec(x), newshape=(nz, n))))

scale = 1e-4
x = np.reshape(x, newshape=(nz, n))
plt.figure()
plt.imshow(np.real(x), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution)")
plt.colorbar()
plt.show()

# Get this by running code once
total_iterations = 30

# #************************************************************
# # Run GMRES variable iterations
# def run_gmres(niter):
#     print("\nRunning GMRES for ", niter, " iterations\n")
#     start_t = time.time()
#     result, exitcode = gmres(
#         A,
#         np.reshape(rhs, newshape=(nz * n, 1)),
#         maxiter=niter,
#         restart=100,
#         callback=make_callback()
#     )
#     print("Exitcode = ", exitcode)
#     end_t = time.time()
#     print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")
#
#     return result
#
# niter_list = [10, 20, 30]
# result_list = []
# for niter in niter_list:
#     x = run_gmres(niter=niter)
#     result_list.append(x)
#
# # Plotting
# fig, axs = plt.subplots(1, 3, sharey=True, figsize=(30, 10))
#
# x = result_list[0]
# x = np.reshape(x, newshape=(nz, n))
# ax = axs[0]
# im0 = ax.imshow(np.real(x), cmap="Greys", vmin=-scale, vmax=scale)
# ax.grid(True)
# ax.set_title("Source", fontname="Times New Roman", fontsize=10)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im0, cax=cax)
#
# x = result_list[1]
# x = np.reshape(x, newshape=(nz, n))
# ax = axs[1]
# im1 = ax.imshow(np.real(x), cmap="Greys", vmin=-scale, vmax=scale)
# ax.grid(True)
# ax.set_title("Source", fontname="Times New Roman", fontsize=10)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im1, cax=cax)
#
# x = result_list[2]
# x = np.reshape(x, newshape=(nz, n))
# ax = axs[2]
# im2 = ax.imshow(np.real(x), cmap="Greys", vmin=-scale, vmax=scale)
# ax.grid(True)
# ax.set_title("Source", fontname="Times New Roman", fontsize=10)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im2, cax=cax)
#
# plt.show()
