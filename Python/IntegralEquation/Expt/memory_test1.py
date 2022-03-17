import multiprocessing as mp
import numpy as np
from . import memory_test0


obj_global = memory_test0.MyClass(n=10)

def do_fft(x):
    y = obj_global.func(x)
    return y

def parallel_fft(array_list_input, array_list_output, obj):

    obj_global.n = obj.n
    obj_global.a = obj.a

    num_items = len(array_list_input)

    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(do_fft, array_list_input)
        pool.close()
        pool.join()

    for i in range(num_items):
        array_list_output[i] = result[i]


if __name__ == "__main__":

    num_items_ = 20
    n_ = 10
    n1_ = 25000

    array_list_input_ = [np.zeros(shape=(n_,), dtype=np.complex64) + 1.0 for _ in range(num_items_)]
    array_list_output_ = [np.zeros(shape=(n_,), dtype=np.complex64) for _ in range(num_items_)]

    obj = memory_test0.MyClass(n=n1_)

    obj.a = np.zeros(shape=(n1_, n1_), dtype=np.float64) + 2.0
    parallel_fft(array_list_input_, array_list_output_, obj)
    print("\nPrinting result:\n")
    print(array_list_output_[0][0])

    obj.a = np.zeros(shape=(n1_, n1_), dtype=np.float64) + 3.0
    parallel_fft(array_list_input_, array_list_output_, obj)
    print("\nPrinting result:\n")
    print(array_list_output_[0][0])

    obj.a = np.zeros(shape=(n1_, n1_), dtype=np.float64) + 4.0
    parallel_fft(array_list_input_, array_list_output_, obj)
    print("\nPrinting result:\n")
    print(array_list_output_[0][0])
