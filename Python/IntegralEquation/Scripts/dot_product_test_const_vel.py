import numpy as np
import time
from ..Solver.ScatteringIntegralConstantVel import TruncatedKernelConstantVel3d as Lipp3d
from ..Solver.ScatteringIntegralConstantVel import TruncatedKernelConstantVel2d as Lipp2d

n_ = 101
k_ = 150.0
precision_ = np.complex128

print("----------------------------------")
print("3D Dot product test results")
print("----------------------------------")
op = Lipp3d(n=n_, k=k_, precision=precision_)

u_ = np.zeros(shape=(n_, n_, n_), dtype=precision_)
v_ = np.zeros(shape=(n_, n_, n_), dtype=precision_)
u_ += np.random.normal(size=(n_, n_, n_))
v_ += np.random.normal(size=(n_, n_, n_))

output_ = u_ * 0

start_t_ = time.time()
op.convolve_kernel(u=u_, output=output_, adj=False)
end_t_ = time.time()
print("Total time to execute convolution (forward): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_1 = np.dot(np.conjugate(np.reshape(v_, newshape=(n_ ** 3,))), np.reshape(output_, newshape=(n_ ** 3,)))

start_t_ = time.time()
op.convolve_kernel(u=v_, output=output_, adj=True)
end_t_ = time.time()
print("Total time to execute convolution (adjoint): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_2 = np.dot(np.conjugate(np.reshape(output_, newshape=(n_ ** 3,))), np.reshape(u_, newshape=(n_ ** 3,)))

print("\nDot product test results:")
print("<v, Au> = ", dot_prod_1)
print("<A*v, u> = ", dot_prod_2)

########################################################
print("\n\n")
print("----------------------------------")
print("2D Dot product test results")
print("----------------------------------")
op = Lipp2d(n=n_, k=k_, precision=precision_)

u_ = np.zeros(shape=(n_, n_), dtype=precision_)
v_ = np.zeros(shape=(n_, n_), dtype=precision_)
u_ += np.random.normal(size=(n_, n_))
v_ += np.random.normal(size=(n_, n_))

output_ = u_ * 0

start_t_ = time.time()
op.convolve_kernel(u=u_, output=output_, adj=False)
end_t_ = time.time()
print("Total time to execute convolution (forward): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_1 = np.dot(np.conjugate(np.reshape(v_, newshape=(n_ ** 2,))), np.reshape(output_, newshape=(n_ ** 2,)))

start_t_ = time.time()
op.convolve_kernel(u=v_, output=output_, adj=True)
end_t_ = time.time()
print("Total time to execute convolution (adjoint): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_2 = np.dot(np.conjugate(np.reshape(output_, newshape=(n_ ** 2,))), np.reshape(u_, newshape=(n_ ** 2,)))

print("\nDot product test results:")
print("<v, Au> = ", dot_prod_1)
print("<A*v, u> = ", dot_prod_2)
