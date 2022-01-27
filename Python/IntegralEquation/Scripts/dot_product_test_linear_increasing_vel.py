import numpy as np
import time
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel2d as Lipp2d
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel3d as Lipp3d

n_ = 101
nz_ = 101
k_ = 60.0
a_ = 0.8
b_ = 1.8
m_ = 1000
precision_ = np.complex128

print("----------------------------------")
print("2D Dot product test results")
print("----------------------------------")
op = Lipp2d(
    n=n_,
    nz=nz_,
    k=k_,
    a=a_,
    b=b_,
    m=m_,
    precision=precision_
)

u_ = np.zeros(shape=(nz_, n_), dtype=precision_)
v_ = np.zeros(shape=(nz_, n_), dtype=precision_)
u_ += np.random.normal(size=(nz_, n_))
v_ += np.random.normal(size=(nz_, n_))

output_ = u_ * 0

start_t_ = time.time()
op.apply_kernel(u=u_, output=output_, adj=False)
end_t_ = time.time()
print("Total time to execute convolution (forward): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_1 = np.dot(np.conjugate(np.reshape(v_, newshape=(nz_ * n_,))), np.reshape(output_, newshape=(nz_ * n_,)))

start_t_ = time.time()
op.apply_kernel(u=v_, output=output_, adj=True)
end_t_ = time.time()
print("Total time to execute convolution (adjoint): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_2 = np.dot(np.conjugate(np.reshape(output_, newshape=(nz_ * n_,))), np.reshape(u_, newshape=(nz_ * n_,)))

print("\nDot product test results:")
print("<v, Au> = ", dot_prod_1)
print("<A*v, u> = ", dot_prod_2)


################################################################
print("\n\n")
print("----------------------------------")
print("3D Dot product test results")
print("----------------------------------")
op = Lipp3d(
    n=n_,
    nz=nz_,
    k=k_,
    a=a_,
    b=b_,
    m=m_,
    precision=precision_
)

u_ = np.zeros(shape=(nz_, n_, n_), dtype=precision_)
v_ = np.zeros(shape=(nz_, n_, n_), dtype=precision_)
u_ += np.random.normal(size=(nz_, n_, n_))
v_ += np.random.normal(size=(nz_, n_, n_))

output_ = u_ * 0

start_t_ = time.time()
op.apply_kernel(u=u_, output=output_, adj=False)
end_t_ = time.time()
print("Total time to execute convolution (forward): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_1 = np.dot(
    np.conjugate(np.reshape(v_, newshape=(nz_ * n_ * n_,))), np.reshape(output_, newshape=(nz_ * n_ * n_,))
)

start_t_ = time.time()
op.apply_kernel(u=v_, output=output_, adj=True)
end_t_ = time.time()
print("Total time to execute convolution (adjoint): ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
dot_prod_2 = np.dot(
    np.conjugate(np.reshape(output_, newshape=(nz_ * n_ * n_,))), np.reshape(u_, newshape=(nz_ * n_ * n_,))
)

print("\nDot product test results:")
print("<v, Au> = ", dot_prod_1)
print("<A*v, u> = ", dot_prod_2)
