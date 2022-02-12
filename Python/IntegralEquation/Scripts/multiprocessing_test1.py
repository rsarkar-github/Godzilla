import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

class SimpleClass1(object):
    def __init__(self, n):
        print("Constructor invoked \n")
        self.arr = np.zeros(shape=(n, n), dtype=np.float32) + 0.5
        self.val = 1.0

    def add(self, a):
        print("Start thread")
        np.sum(self.arr)
        for _ in range(10):
            for _ in range(100000000):
                b = 1.0
        print(self.val + a)

    def printing(self):
        print(self.arr[0, 0])

class SimpleClass2(object):
    def __init__(self, n):
        print("Constructor invoked \n")
        self.arr = np.zeros(shape=(n, n), dtype=np.float32) + 0.5
        self.val = 1.0

    def add(self, a):
        print("Start thread")
        for _ in range(100):
            self.arr += 1.0
        print(self.val + a)
        print(self.arr[0, 0])

    def printing(self):
        print(self.arr[0, 0])

n = 20000

inst = SimpleClass1(n)
# inst = SimpleClass2(n)

def func(arr):
    inst.add(arr)

if __name__ == '__main__':

    mp.set_start_method('fork')
    print("CPU count = ", mp.cpu_count(), "\n")

    ntimes = 8

    arglist = [(i) for i in range(ntimes)]

    with Pool(mp.cpu_count()) as pool:
        pool.map(func, arglist)
        pool.close()
        pool.join()

    inst.printing()
