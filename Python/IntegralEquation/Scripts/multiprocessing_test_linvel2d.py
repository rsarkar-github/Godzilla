import numpy as np
import multiprocessing as mp
from multiprocessing import Process
from ..Solver.ScatteringIntegralLinearIncreasingVel import TruncatedKernelLinearIncreasingVel2d as Lipp2d


def func(obj, input, output):
    obj.apply_kernel(input, output)

if __name__ == "__main__":

    n_ = 101
    nz_ = 101
    k_ = 100.0
    a_ = 0.8
    b_ = 1.8
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

    u_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    u_[int(nz_ / 8), int(n_ / 2)] = 1.0
    u1_ = u_ * 1.0
    output_ = u_ * 0
    output1_ = u_ * 0

    # plist = []
    # p1 = Process(target=func, args=[op, u_, output_])
    # plist.append(p1)
    #
    # p2 = Process(target=func, args=[op, u1_, output1_])
    # plist.append(p2)

    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap(func, [(op, u_, output_), (op, u1_, output1_)])

    # for p in plist:
    #     print("Process ", p.name, "starting\n")
    #     p.start()
    #
    # for p in plist:
    #     p.join()
    #     print("Process ", p.name, "done\n")

    print("All Close check : ", np.allclose(output_, output1_))
    print(output_[0, 0])
    print(output1_[0, 0])
