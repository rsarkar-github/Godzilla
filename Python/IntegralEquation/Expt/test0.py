import numpy as np

class MyClass(object):
    def __init__(self):
        self._a = 1.0

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    def func(self, x):
        print("\nself._a = ", self._a)
        return np.fft.fftn(x) + self._a
