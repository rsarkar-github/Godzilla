import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import splu
import time
import matplotlib.pyplot as plt
from ..Solver import HelmholtzOperators


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
precision = np.complex128

# Create linearly varying background
vel = np.zeros(shape=(nz, n), dtype=np.float64)
for i in range(nz):
    vel[i, :] = alpha * (a + i * hz)
# vel = np.zeros(shape=(nz, n), dtype=np.float64) + 3.0

# Plot velocity
plt.figure()
plt.imshow(vel, extent=[xmin, xmax, b, a], cmap="jet", vmin=2.0, vmax=3.0)
plt.grid(True)
plt.title("Background Velocity")
plt.colorbar()
plt.show()

# Create Gaussian perturbation
pert_gaussian = np.zeros(shape=(nz, n), dtype=np.float64)
pert_gaussian[int((nz - 1) / 2), int((n - 1) / 2)] = 1000.0
pert_gaussian = gaussian_filter(pert_gaussian, sigma=10)

# Plot gaussian perturbation
plt.figure()
plt.imshow(pert_gaussian, extent=[xmin, xmax, b, a], cmap="Greys", vmin=0, vmax=0.01)
plt.grid(True)
plt.title("Perturbation")
plt.colorbar()
plt.show()

# Create 2D velocity and perturbation fields
xgrid = np.linspace(start=xmin, stop=xmax, num=n, endpoint=True)
total_vel = vel + pert_gaussian
# total_vel = vel + 0.0

# Plot total velocity
plt.figure()
plt.imshow(total_vel, extent=[xmin, xmax, b, a], cmap="jet", vmin=2.0, vmax=3.0)
plt.grid(True)
plt.title("Total Velocity")
plt.colorbar()
plt.show()

# Source
p = 0.0
q = a + (b - a) / 3
# q = a + (b - a) / 2
sigma = 0.01
zgrid = np.linspace(start=a, stop=b, num=nz, endpoint=True)
z, x1 = np.meshgrid(zgrid, xgrid / 1, indexing="ij")
distsq = (z - q) ** 2 + (x1 - p) ** 2
f = np.exp(-0.5 * distsq / (sigma ** 2))

# Plot source
plt.figure()
plt.imshow(f, extent=[xmin, xmax, b, a], cmap="jet")
plt.grid(True)
plt.title("Source")
plt.colorbar()
plt.show()

# Create Helmholtz matrix
mat = HelmholtzOperators.create_helmholtz2d_matrix(
    a1=xmax-xmin,
    a2=b-a,
    pad1=20,
    pad2=20,
    omega=omega,
    precision=precision,
    vel=total_vel,
    pml_damping=50.0,
    adj=False,
    warnings=True
)

start_t = time.time()
mat_lu = splu(mat)
end_t = time.time()
print("\nTotal time to LU factorize: ", "{:4.2f}".format(end_t - start_t), " s \n")

start_t = time.time()
x1 = mat_lu.solve(np.reshape(f, newshape=(nz * n, 1)))
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

mat = HelmholtzOperators.create_helmholtz2d_matrix(
    a1=xmax-xmin,
    a2=b-a,
    pad1=20,
    pad2=20,
    omega=omega,
    precision=precision,
    vel=total_vel,
    pml_damping=500.0,
    adj=False,
    warnings=True
)

start_t = time.time()
mat_lu = splu(mat)
end_t = time.time()
print("\nTotal time to LU factorize: ", "{:4.2f}".format(end_t - start_t), " s \n")

start_t = time.time()
x2 = mat_lu.solve(np.reshape(f, newshape=(nz * n, 1)))
end_t = time.time()
print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

scale = 1e-5
x = np.reshape(x1-x2, newshape=(nz, n))
plt.figure()
plt.imshow(np.real(x), extent=[xmin, xmax, b, a], cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real (Solution)")
plt.colorbar()
plt.show()
