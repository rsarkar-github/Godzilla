# Running this script on Windows and Linux has different behavior

import multiprocessing as mp
import numpy as np
from . import test0


obj_global = test0.MyClass()

def do_fft(x):
    y = obj_global.func(x)
    return y

def parallel_fft(array_list_input, array_list_output, obj):

    obj_global.a = obj.a
    num_items = len(array_list_input)

    print("\nValue of obj_global.a before entering parallel section = ", obj_global.a)
    print("\n")

    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(do_fft, array_list_input)
        pool.close()
        pool.join()

    for i in range(num_items):
        array_list_output[i] = result[i]

    print("\nValue of obj_global.a after exiting parallel section = ", obj_global.a)
    print("\n")


if __name__ == "__main__":

    num_items_ = 2
    n_ = 10

    array_list_input_ = [np.zeros(shape=(n_,), dtype=np.complex64) + 1.0 for _ in range(num_items_)]
    array_list_output_ = [np.zeros(shape=(n_,), dtype=np.complex64) for _ in range(num_items_)]

    obj = test0.MyClass()
    obj.a = 2.0

    parallel_fft(array_list_input_, array_list_output_, obj)

    print("Printing result:\n")
    print(array_list_output_)
