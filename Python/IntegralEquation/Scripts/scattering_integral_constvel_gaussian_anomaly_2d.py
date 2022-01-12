import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, gmres, lsmr
import time
import matplotlib.pyplot as plt
from ..Solver.ScatteringIntegralConstantVel import TruncatedKernelConstantVel2d as Lipp2d

# Need to make sure n = nz
n = 101
nz = 101

# Need to make sure (b- a) =  (xmax - xmin)
a = 4
b = 9
xmin = -2.5
xmax = 2.5
hz = (b - a) / (nz - 1)
hx = (xmax - xmin) / (n - 1)
c = 2.5

# Create linearly varying background
vel = np.zeros(shape=(nz, n), dtype=np.float64)
for i in range(nz):
    vel[i, :] = c

# Plot velocity
fig = plt.figure()
plt.imshow(vel, cmap="jet", vmin=2.0, vmax=4.0)
plt.grid(True)
plt.title("Velocity")
plt.colorbar()
plt.show()

# Create Gaussian perturbation
pert_gaussian = np.zeros(shape=(nz, n), dtype=np.float64)
pert_gaussian[int((nz - 1) / 2), int((n - 1) / 2)] = 1500.0
pert_gaussian = gaussian_filter(pert_gaussian, sigma=10)

# # Plot gaussian perturbation
# fig = plt.figure()
# plt.imshow(pert_gaussian, cmap="Greys", vmin=0, vmax=0.01)
# plt.grid(True)
# plt.title("Perturbation")
# plt.colorbar()
# plt.show()

# Create 3D velocity and perturbation fields using chi cutoff
xgrid = np.linspace(start=xmin, stop=xmax, num=n, endpoint=True)
total_vel = vel + pert_gaussian

# Plot velocity
fig = plt.figure()
plt.imshow(total_vel, cmap="jet", vmin=2.0, vmax=4.0)
plt.grid(True)
plt.title("Total Velocity")
plt.colorbar()
plt.show()

# Scaled problem
scale = np.abs(xmax - xmin)
a = a / scale
b = b / scale
omega = 12 * np.pi
k = omega * scale / c
psi = (1.0 - c ** 2.0 / (total_vel ** 2))
precision = np.complex64

# Plot psi
fig = plt.figure()
plt.imshow(psi, cmap="Greys")
plt.grid(True)
plt.title("Psi")
plt.colorbar()
plt.show()

psi = psi.astype(precision)


# Source
p = 0.0
q = a + (b - a) / 10.0
sigma = 0.025
zgrid = np.linspace(start=a, stop=b, num=nz, endpoint=True)
z, x1 = np.meshgrid(zgrid, xgrid / 5, indexing="ij")
distsq = (z - q) ** 2 + (x1 - p) ** 2
f = np.exp(-0.5 * distsq / (sigma ** 2))

# # Plot source
# fig = plt.figure()
# plt.imshow(f, cmap="jet")
# plt.grid(True)
# plt.title("Source")
# plt.colorbar()
# plt.show()

# Initialize operator
op = Lipp2d(
    n=n,
    k=k,
    precision=precision
)

# Create rhs
f = f.astype(precision)
rhs = np.zeros((nz, n), dtype=precision)
start_t = time.time()
op.convolve_kernel(u=f, output=rhs)
end_t = time.time()
print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")
print("Finished rhs computation\n")

# scale = 1e-5
# fig = plt.figure()
# # np.savez(file="G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/alpha0.5.npz", args=rhs)
# plt.imshow(np.real(rhs), cmap="Greys", vmin=-scale, vmax=scale)
# plt.grid(True)
# plt.title("Real")
# plt.colorbar()
# plt.show()


# Define linear operator object
def func_matvec(v):
    v = np.reshape(v, newshape=(nz, n))
    u = v * 0
    op.convolve_kernel(u=v*psi, output=u, adj=False, add=False)
    return np.reshape(v + (k ** 2) * u, newshape=(nz * n, 1))

A = LinearOperator(shape=(nz * n, nz * n), matvec=func_matvec, dtype=precision)

# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[])

    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print(closure_variables["counter"], residuals)
    return callback


# # Load initial solution
# x0 = np.load("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/linvel_gaussian_sol1x.npz")["arr_0"]
# x0 = np.reshape(x0, newshape=(nz * (n ** 2), 1))

# Run gmres
start_t = time.time()
x, exitCode = gmres(
    A,
    np.reshape(rhs, newshape=(nz * n, 1)),
    maxiter=300,
    restart=100,
    callback=make_callback()
)
# x = lsmr(
#     A,
#     np.reshape(rhs, newshape=(nz * n, 1))
# )[:1]
# print(exitCode)
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")
print("Residual norm = ", np.linalg.norm(rhs - np.reshape(A.matvec(x), newshape=(nz, n))))

scale = 1e-5
x = np.reshape(x, newshape=(nz, n))
np.savez("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/constvel_gaussian_sol_test1_c2.5.npz", x)
fig = plt.figure()
plt.imshow(np.real(x), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real")
plt.colorbar()
plt.show()
