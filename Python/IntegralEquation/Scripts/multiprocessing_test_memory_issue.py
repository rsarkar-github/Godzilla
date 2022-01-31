######################################################################################
# Test memory issue
#-------------------------------------------------------------------------------------

from multiprocessing import Pool
import  multiprocessing as mp
import numpy as np

n = 5000

def f(x):
    for _ in range(10):
        arr2 = np.zeros(shape=(n, n), dtype=np.float32)
        # arr2 = np.zeros(shape=(n, n), dtype=np.float32) + 1.0
    print("Starting x = ", x)
    return x * x

def f1(x):
    for _ in range(10):
        arr2 = np.zeros(shape=(n, n), dtype=np.float32)
        # arr2 = arr2 + 1.0
        # arr2 = np.zeros(shape=(n, n), dtype=np.float32) + 1.0
    print("Starting x1 = ", x[0], " , x2 = ", x[1])
    return x[0] * x[1]

def f2(x, y):
    for _ in range(10):
        arr2 = np.zeros(shape=(n, n), dtype=np.float32)
        # arr2 = np.zeros(shape=(n, n), dtype=np.float32) + 1.0
    print("Starting x = ", x, " , y = ", y)
    return x * y


if __name__ == '__main__':
    with Pool(mp.cpu_count()) as p:
        print(p.map(f1, [(i, i) for i in range(50)]), "\n")
