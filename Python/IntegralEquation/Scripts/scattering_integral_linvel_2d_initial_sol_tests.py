import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, gmres
import time
import matplotlib.pyplot as plt
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel2d as Lipp2d


n = 101
nz = 101
a = 4
b = 5
xmin = -0.5
xmax = 0.5
hz = (b - a) / (nz - 1)
hx = (xmax - xmin) / (n - 1)
alpha = 0.5

omega = 30 * 2* np.pi
k = omega / alpha
m = 500
precision = np.complex64

# Set GMRES tol
rtol = 1e-6

# Create linearly varying background
vel = np.zeros(shape=(nz, n), dtype=np.float64)
for i in range(nz):
    vel[i, :] = alpha * (a + i * hz)

# Plot velocity
plt.figure()
plt.imshow(vel, cmap="jet", vmin=2.0, vmax=3.0)
plt.grid(True)
plt.title("Background Velocity")
plt.colorbar()
plt.show()

# Create Gaussian perturbation
pert_gaussian = np.zeros(shape=(nz, n), dtype=np.float64)
pert_gaussian[int((nz - 1) / 2), int((n - 1) / 2)] = 1000.0
pert_gaussian = gaussian_filter(pert_gaussian, sigma=10)

# Create 2D velocity and perturbation fields
xgrid = np.linspace(start=xmin, stop=xmax, num=n, endpoint=True)
total_vel = vel + pert_gaussian

# Plot total velocity
plt.figure()
plt.imshow(total_vel, cmap="jet", vmin=2.0, vmax=3.0)
plt.grid(True)
plt.title("Total Velocity")
plt.colorbar()
plt.show()


# Calculate psi
psi = (alpha ** 2) * (1.0 / (vel ** 2) - 1.0 / (total_vel ** 2))

# # Plot psi
# plt.figure()
# plt.imshow(psi, cmap="Greys")
# plt.grid(True)
# plt.title("Psi")
# plt.colorbar()
# plt.show()

psi = psi.astype(precision)


# Source
p = 0.0
q = a + (b - a) / 10.0
sigma = 0.025
zgrid = np.linspace(start=a, stop=b, num=nz, endpoint=True)
z, x1 = np.meshgrid(zgrid, xgrid / 1, indexing="ij")
distsq = (z - q) ** 2 + (x1 - p) ** 2
f = np.exp(-0.5 * distsq / (sigma ** 2))

# # Plot source
# plt.figure()
# plt.imshow(f, cmap="jet")
# plt.grid(True)
# plt.title("Source")
# plt.colorbar()
# plt.show()

# Initialize operator
op = Lipp2d(
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
rhs = np.zeros((nz, n), dtype=precision)
start_t = time.time()
op.apply_kernel(u=f, output=rhs)
end_t = time.time()
print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")
print("Norm of rhs = ", np.linalg.norm(rhs), "\n")
print("Finished rhs computation\n")

# scale = 1e-5
# fig = plt.figure()
# plt.imshow(np.real(rhs), cmap="Greys", vmin=-scale, vmax=scale)
# plt.grid(True)
# plt.title("Real (rhs)")
# plt.colorbar()
# plt.show()

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

# Run gmres
print("\n----------------------------------------------")
print("\nStarting GMRES to compute initial solution")
print("\n----------------------------------------------\n")
start_t = time.time()
x0, exitCode = gmres(
    A,
    np.reshape(rhs, newshape=(nz * n, 1)),
    maxiter=300,
    restart=100,
    tol=rtol,
    atol=0.0,
    callback=make_callback()
)
print("\nLinear solver exitcode:", exitCode)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

# Plot solution at perturbation
scale = 1e-5
x0 = np.reshape(x0, newshape=(nz, n))
plt.figure()
plt.imshow(np.real(x0), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution)")
plt.colorbar()
plt.show()

# Reset psi
pert_fac = 1.1
print("\nPerturbation scaling = ", pert_fac, "\n")
pert_gaussian *= pert_fac
psi *= 0
psi += (alpha ** 2) * (1.0 / (vel ** 2) - 1.0 / ((vel + pert_gaussian) ** 2))

# Calculate new rhs
rhs1 = np.zeros((nz, n), dtype=precision)
op.apply_kernel(u=psi*x0, output=rhs1)
rhs1 = rhs - x0 + (k ** 2) * rhs1

# Run gmres
print("\n----------------------------------------------")
print("\nStarting GMRES with initial solution")
print("\n----------------------------------------------\n")

print("Norm of new rhs = ", np.linalg.norm(rhs1), "\n")
print("Tolerance for solver = ", rtol * np.linalg.norm(rhs) / np.linalg.norm(rhs1))
start_t = time.time()
x, exitCode = gmres(
    A,
    np.reshape(rhs1, newshape=(nz * n, 1)),
    maxiter=300,
    restart=100,
    tol=rtol * np.linalg.norm(rhs) / np.linalg.norm(rhs1),
    atol=0.0,
    callback=make_callback()
)
print("\nLinear solver exitcode:", exitCode)
end_t = time.time()
print("Total time to solve (with initial solution): ", "{:4.2f}".format(end_t - start_t), " s \n")

# Total solution
xtotal = x0 + np.reshape(x, newshape=(nz, n))

# Plot solution
scale = 1e-5
plt.figure()
plt.imshow(np.real(xtotal), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution)")
plt.colorbar()
plt.show()

# Run gmres
print("\n----------------------------------------------")
print("\nStarting GMRES with no initial solution")
print("\n----------------------------------------------\n")

print("Tolerance for solver = ", rtol)
start_t = time.time()
x, exitCode = gmres(
    A,
    np.reshape(rhs, newshape=(nz * n, 1)),
    maxiter=300,
    restart=100,
    tol=rtol,
    atol=0.0,
    callback=make_callback()
)
print("\nLinear solver exitcode:", exitCode)
end_t = time.time()
print("Total time to solve (without initial solution): ", "{:4.2f}".format(end_t - start_t), " s \n")

print("\nDifference in norm of two solutions = ", np.linalg.norm(xtotal - np.reshape(x, newshape=(nz, n))))

# Plot solution
scale = 1e-5
plt.figure()
plt.imshow(np.real(np.reshape(x, newshape=(nz, n))), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution)")
plt.colorbar()
plt.show()
