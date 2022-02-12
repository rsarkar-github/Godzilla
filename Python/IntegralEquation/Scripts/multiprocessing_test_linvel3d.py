import numpy as np
import multiprocessing as mp
import time
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel3d as Lipp3d


n_ = 101
nz_ = 101
k_ = 100.0
a_ = 0.8
b_ = 1.8
m_ = 1000
precision_ = np.complex128

op = Lipp3d(
    n=n_,
    nz=nz_,
    k=k_,
    a=a_,
    b=b_,
    m=m_,
    precision=precision_
)

def func(input, output):
    print("Starting thread")
    start_t_ = time.time()
    op.apply_kernel(input, output)
    end_t_ = time.time()
    print("Total time to execute convolution: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")

if __name__ == "__main__":

    u_ = np.zeros(shape=(nz_, n_, n_), dtype=precision_)
    u_[int(nz_ / 8), int(n_ / 2), int(n_ / 2)] = 1.0
    output_ = u_ * 0

    ntimes = 20
    arglist = [(u_ * 1.0, output_ * 1.0) for _ in range(ntimes)]

    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap(func, arglist)
