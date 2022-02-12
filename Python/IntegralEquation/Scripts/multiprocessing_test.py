import numpy as np
import numba
import multiprocessing as mp
from multiprocessing import Pool

class SimpleClass(object):
    def __init__(self, n):
        print("Constructor invoked \n")
        self.val = 1.0

    def add(self, a):
        print("Start thread")
        for _ in range(50):
            arr2 = np.zeros(shape=(n, n), dtype=np.float32) + 1.0
        print(self.val + a)

class SimpleClass1(object):
    def __init__(self, n):
        print("Constructor invoked \n")
        self.arr = np.zeros(shape=(n, n), dtype=np.float32) + 0.5
        self.val = 1.0

    def add(self, a):
        print("Start thread")
        for _ in range(10):
            for _ in range(100000000):
                b = 1.0
        print(self.val + a)

class SimpleClass2(object):
    def __init__(self, n):
        print("Constructor invoked \n")
        self.arr = np.zeros(shape=(n, n), dtype=np.float32)
        self.val = 1.0

    def add(self, a):
        print("Start thread")
        for _ in range(100):
            self.func_sum(self.arr)
            np.sum(self.arr)
        print(self.val + a)

    @staticmethod
    @numba.jit(nopython=True)
    def func_sum(arr):
        sum = 0.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sum = sum + arr[i, j]
        return sum

class SimpleClass3(object):
    def __init__(self, n):
        print("Constructor invoked \n")
        self.arr = np.zeros(shape=(n, n), dtype=np.float32) + 0.5
        self.val = 1.0

    def add(self, a):
        print("Start thread")
        for _ in range(100):
            self.arr += 0.0
        print(self.val + a)

def func(obj, arr):
    obj.add(arr)

if __name__ == '__main__':

    mp.set_start_method('fork')
    print("CPU count = ", mp.cpu_count(), "\n")

    n = 20000
    ntimes = 8

    # inst = SimpleClass(n)  #===> behaves as expected, creates copies
    # inst = SimpleClass1(n)
    # inst = SimpleClass2(n)
    inst = SimpleClass3(n)

    arglist = [(inst, i) for i in range(ntimes)]

    with Pool(mp.cpu_count()) as pool:
        pool.starmap(func, arglist)
        # for a in arglist:
        #     pool.apply_async(func, (a))
        pool.close()
        pool.join()
