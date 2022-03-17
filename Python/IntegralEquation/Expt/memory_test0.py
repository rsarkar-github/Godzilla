import numpy as np

class MyClass(object):
    def __init__(self, n):
        self._n = n
        self._a = np.zeros(shape=(n, n), dtype=np.float64)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n
        self._a = np.zeros(shape=(n, n), dtype=np.float64)

    def func(self, x):

        y = 0
        for _ in range(10):
            y = np.fft.fftn(x) + np.sum(self._a)

        return y
