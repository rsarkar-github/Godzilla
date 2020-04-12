import numpy as np
import time
from numba import jit
import threading


@jit(nopython=True, cache=True)
def go_fast(a1):  # Function is compiled and runs in machine code
    trace = 0
    for i in range(a1.shape[0]):
        trace += np.tanh(a1[i, i])
    return a1 + trace


def add_vectors(x1, y1):
    return x1 + y1


x = np.arange(100000000).reshape(10000, 10000)

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
for _ in range(5):
    start = time.time()
    go_fast(x)
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))

print("\n\n")
a = x + 1
b = x + 2

start = time.time()
add_vectors(a, b)
add_vectors(a, b)
add_vectors(a, b)
add_vectors(a, b)
end = time.time()
print("Elapsed = %s" % (end - start))

start = time.time()
t1 = threading.Thread(target=add_vectors, args=(a, b))
t2 = threading.Thread(target=add_vectors, args=(a, b))
t3 = threading.Thread(target=add_vectors, args=(a, b))
t4 = threading.Thread(target=add_vectors, args=(a, b))
t1.start()
t2.start()
t3.start()
t4.start()
t1.join()
t2.join()
t3.join()
t4.join()
end = time.time()
print("Elapsed = %s" % (end - start))
