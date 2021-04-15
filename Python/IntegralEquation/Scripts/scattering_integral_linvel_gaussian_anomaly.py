import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, gmres
import time
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel3d as Lipp3d


n = 61
nz = 61
vel_salt = 4.5
vel_water = 1.5
a = 4
b = 9
xmin = -2.5
xmax = 2.5
hz = (b - a) / (nz - 1)
hx = (xmax - xmin) / (n - 1)
alpha = 0.5

# Create linearly varying background
vel = np.zeros(shape=(nz, n), dtype=np.float32)
for i in range(nz):
    vel[i, :] = alpha * (a + i * hz)

# Create Gaussian perturbation
pert_gaussian = np.zeros(shape=(nz, n), dtype=np.float32)
pert_gaussian[int((nz - 1) / 2), int((n - 1) / 2)] = 700.0
pert_gaussian = gaussian_filter(pert_gaussian, sigma=10)

# Create 3D velocity and perturbation fields using chi cutoff
xgrid = np.linspace(start=xmin, stop=xmax, num=n, endpoint=True)
chi = xgrid * 0
p1 = 5/3
p2 = 5/4
for i in range(n):
    if np.abs(xgrid[i]) < p1:
        if np.abs(xgrid[i]) <= p2:
            chi[i] = 1.0
        else:
            chi[i] = 1.0 - np.exp(- 1.0 / (xgrid[i]**2 - p2**2)) / \
                   (np.exp(- 1.0 / (xgrid[i]**2 - p2**2)) + np.exp(- 1.0 / (p1 ** 2 - xgrid[i]**2)))

vel3d = np.zeros(shape=(nz, n, n), dtype=np.float32) + np.reshape(vel, newshape=(nz, n, 1))
total_vel3d = vel3d + np.reshape(pert_gaussian, newshape=(nz, n, 1)) * np.reshape(chi, newshape=(1, 1, n))

# Scaled problem
scale = np.abs(xmax - xmin)
alpha = alpha
a = a / scale
b = b / scale
omega = 10 * np.pi
k = (omega / alpha) ** 2
psi = (1.0 / (vel3d ** 2) - 1.0 / (total_vel3d ** 2)) * (scale ** 2)
m = 200
precision = np.complex64
psi = psi.astype(precision)

# Source
p = 0.0
q = a + (b - a) / 10.0
sigma = 0.025
zgrid = np.linspace(start=a, stop=b, num=nz, endpoint=True)
z, x1, x2 = np.meshgrid(zgrid, xgrid / 5, xgrid / 5, indexing="ij")
distsq = (z - q) ** 2 + (x1 - p) ** 2 + x2 ** 2
f = np.exp(-0.5 * distsq / (sigma ** 2))

# Initialize operator
op = Lipp3d(
    n=n,
    nz=nz,
    k=k,
    a=a,
    b=b,
    m=m,
    precision=precision
)

# Create rhs
f = f.astype(precision)
rhs = np.zeros((nz, n, n), dtype=precision)
start_t = time.time()
op.apply_kernel(u=f, output=rhs)
end_t = time.time()
print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")


# Define linear operator object
def func_matvec(v):
    v = np.reshape(v, newshape=(nz, n, n))
    u = v * 0
    op.apply_kernel(u=v*psi, output=u)
    return np.reshape(v - (omega ** 2) * u, newshape=(nz * (n**2), 1))


A = LinearOperator(shape=(nz * (n**2), nz * (n**2)), matvec=func_matvec, dtype=precision)


# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[])

    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print(closure_variables["counter"], residuals)
    return callback


# Run gmres
start_t = time.time()
x, exitCode = gmres(
    A,
    np.reshape(rhs, newshape=(nz * (n**2), 1)),
    maxiter=200,
    restart=20,
    callback=make_callback()
)
print(exitCode)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")
print("Residual norm = ", np.linalg.norm(rhs - np.reshape(A.matvec(x), newshape=(nz, n, n))))

x = np.reshape(x, newshape=(nz, n, n))
np.savez("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/linvel_gaussian_sol.npz", x)
