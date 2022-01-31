import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

class SimpleClass(object):
    def __init__(self, n):
        self.val = 1.0

    def add(self, a):
        for _ in range(50):
            arr2 = np.zeros(shape=(20000, 20000), dtype=np.float32) + 1.0
        print(self.val + a)

class SimpleClass1(object):
    def __init__(self, n):
        self.arr = np.zeros(shape=(20000, 20000), dtype=np.float32) + 0.5
        self.val = 1.0

    def add(self, a):
        for _ in range(10):
            for _ in range(100000000):
                b = 1.0
        print(self.val + a)

class SimpleClass2(object):
    def __init__(self, n):
        self.arr = np.zeros(shape=(20000, 20000), dtype=np.float32) + 0.5
        self.val = 1.0

    def add(self, a):
        for _ in range(10):
            np.sum(self.arr)
        print(self.val + a)

class SimpleClass3(object):
    def __init__(self, n):
        self.arr = np.zeros(shape=(20000, 20000), dtype=np.float32) + 0.5
        self.val = 1.0
        print("Constructor")

    def add(self, a):
        for _ in range(10):
            self.arr += 0.0
        print(self.val + a)

def func(obj, arr):
    obj.add(arr)

if __name__ == '__main__':

    mp.set_start_method('fork')
    print("CPU count = ", mp.cpu_count())

    n = 3000
    ntimes = 20

    # inst = SimpleClass(n)
    # inst = SimpleClass1(n)
    # inst = SimpleClass2(n)
    inst = SimpleClass3(n)

    pool = mp.Pool(mp.cpu_count())
    arglist = [(inst, i) for i in range(ntimes)]
    result = pool.starmap(func, arglist)

