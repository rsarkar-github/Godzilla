import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel2d as Lipp2d


n_ = 101
nz_ = 101
k_ = 300.0
a_ = 4.0
b_ = 5.0
alpha_ = 0.5
m_ = 1000
precision_ = np.complex64

op = Lipp2d(
    n=n_,
    nz=nz_,
    k=k_,
    a=a_,
    b=b_,
    m=m_,
    precision=precision_
)

psi = np.zeros(shape=(nz_, n_), dtype=np.float32)
rtol = 1e-6

def func_solve(source, thread_num):
    """
    :param source: rhs of LSE without application of the kernel
    :param thread_num: thread number
    :return: solution of LSE
    """

    print("Starting thread ", thread_num)
    start_t_ = time.time()

    # Create RHS of LSE
    rhs = source * 0
    op.apply_kernel(u=source, output=rhs)

    # Create Linear Operator for GMRES
    def func_matvec(v):
        v = np.reshape(v, newshape=(nz_, n_))
        u = v * 0
        op.apply_kernel(u=v * psi, output=u, adj=False, add=False)
        return np.reshape(v - (k_ ** 2) * u, newshape=(nz_ * n_, 1))

    linop = LinearOperator(shape=(nz_ * n_, nz_ * n_), matvec=func_matvec, dtype=precision_)

    # Callback generator
    msg_list = ["GMRES on thread " + str(thread_num) + "\n\n"]
    def make_callback():
        closure_variables = dict(counter=0, residuals=[])
        def callback(residuals):
            closure_variables["counter"] += 1
            closure_variables["residuals"].append(residuals)
            msg_list.append("Iteration = " + str(closure_variables["counter"])
                            + ", Residual = " + str(residuals) + "\n")
        return callback

    output, exitcode = gmres(
        linop,
        np.reshape(rhs, newshape=(nz_ * n_, 1)),
        maxiter=300,
        restart=100,
        tol=rtol,
        atol=0.0,
        callback=make_callback()
    )
    end_t_ = time.time()
    msg_list.append("\nLinear solver exitcode:" + str(exitcode))
    msg_list.append("Total time for GMRES: " + "{:4.2f}".format(end_t_ - start_t_) + " s \n")

    print("Finished thread ", thread_num)
    return (np.reshape(output, newshape=(nz_, n_)), rhs, msg_list)

if __name__ == "__main__":

    xmin = -0.5
    xmax = 0.5
    hz = (b_ - a_) / (nz_ - 1)
    hx = (xmax - xmin) / (n_ - 1)

    # Create linearly varying background
    vel = np.zeros(shape=(nz_, n_), dtype=np.float32)
    for i in range(nz_):
        vel[i, :] = alpha_ * (a_ + i * hz)

    # Plot velocity
    plt.figure()
    plt.imshow(vel, cmap="jet", vmin=2.0, vmax=3.0)
    plt.grid(True)
    plt.title("Background Velocity")
    plt.colorbar()
    plt.show()

    # Create Gaussian perturbation
    pert_gaussian = np.zeros(shape=(nz_, n_), dtype=np.float32)
    pert_gaussian[int((nz_ - 1) / 2), int((n_ - 1) / 2)] = 1000.0
    pert_gaussian = gaussian_filter(pert_gaussian, sigma=10)

    # Create 2D velocity and perturbation fields and psi
    xgrid = np.linspace(start=xmin, stop=xmax, num=n_, endpoint=True)
    total_vel = vel + pert_gaussian
    psi += (alpha_ ** 2) * (1.0 / (vel ** 2) - 1.0 / (total_vel ** 2))

    # Plot total velocity
    plt.figure()
    plt.imshow(total_vel, cmap="jet", vmin=2.0, vmax=3.0)
    plt.grid(True)
    plt.title("Total Velocity")
    plt.colorbar()
    plt.show()

    # Source
    p = 0.0
    q = a_ + (b_ - a_) / 10.0
    sigma = 0.025
    zgrid = np.linspace(start=a_, stop=b_, num=nz_, endpoint=True)
    z, x1 = np.meshgrid(zgrid, xgrid / 1, indexing="ij")
    distsq = (z - q) ** 2 + (x1 - p) ** 2
    f = np.exp(-0.5 * distsq / (sigma ** 2))

    # u_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    # u_[int(nz_ / 8), int(n_ / 2)] = 1.0
    f = f.astype(precision_)

    nthreads = 20
    arglist = [(f * 1.0, thread_num) for thread_num in range(nthreads)]

    pool = mp.Pool(min(nthreads, mp.cpu_count()))
    output_result_list = pool.starmap(func_solve, arglist)

    # Print messages
    for num_thread in range(nthreads):
        msg = output_result_list[num_thread][2]
        print("\n\n")
        print("----------------------------------------------\n")
        print("Printing message logs from thread ", num_thread)
        print("----------------------------------------------\n")
        for item in msg:
            print(item)

    # Plot solution
    scale = 1e-4
    plt.figure()
    plt.imshow(np.real(output_result_list[0][0]), cmap="Greys", vmin=-scale, vmax=scale)
    plt.grid(True)
    plt.title("Real (Solution)")
    plt.colorbar()
    plt.show()

    # Plot solution
    scale = 1e-4
    plt.figure()
    plt.imshow(np.real(output_result_list[0][1]), cmap="Greys", vmin=-scale, vmax=scale)
    plt.grid(True)
    plt.title("Real (Solution)")
    plt.colorbar()
    plt.show()
