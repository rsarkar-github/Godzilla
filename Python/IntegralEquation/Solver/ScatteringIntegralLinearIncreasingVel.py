import numpy as np
import scipy.special as sp
import time
import numba
import numba_scipy
from ...Utilities import TypeChecker
import matplotlib.pyplot as plt


class TruncatedKernelLinearIncreasingVel3d:

    def __init__(self, n, nz, k, a, b, m, precision):
        """
        :param n: The field will have shape n x n x nz, with n odd. This means that the unit cube
        [-0.5, 0.5]^2 is gridded into n^2 nz points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param k: The Helmholtz equation reads (lap + k^2 / z^2)u = f.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param m: Number of points to use in Riemann sum calculator for FT of truncated kernel.
        :param precision: np.complex64 or np.complex128
        """

        print("\n\nInitializing the class")

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check(x=nz, expected_type=(int,))
        if nz < 2:
            raise ValueError("n must be an integer >= 2")

        TypeChecker.check_float_positive(k)
        TypeChecker.check_float_positive(a)
        TypeChecker.check_float_strict_lower_bound(x=b, lb=a)

        TypeChecker.check_int_positive(m)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        self._n = n
        self._nz = nz
        self._k = k
        self._a = a
        self._b = b
        self._m = m
        self._precision = precision

        # Run class initializer
        self.__initialize_class()

    def apply_kernel(self, u, output):
        """
        :param u: 3d numpy array (must be n x n x nz dimensions with n odd).
        :param output: 3d numpy array (same dimension as u). Assumed to be zeros.
        """

        # Check types of input
        if u.dtype != self._precision or output.dtype != self._precision:
            raise TypeError("Types of 'u' and 'output' must match that of class: ", self._precision)

        # Check dimensions
        if u.shape != (self._nz, self._n, self._n) or output.shape != (self._nz, self._n, self._n):
            raise ValueError("Shapes of 'u' and 'output' must be (nz, n, n) with nz = ", self._nz, " and n = ", self._n)

        # Copy u into self._temparray
        self._temparray[:, self._start_index:(self._end_index + 1), self._start_index:(self._end_index + 1)] = u

        # Compute Fourier transform along first 2 axis
        temparray = np.fft.fftn(np.fft.fftshift(self._temparray, axes=(1, 2)), axes=(1, 2))

        # Do the following for each z slice
        # 1. Multiply with Fourier transform of Truncated Kernel Green's function slice for that z
        # 2. Compute weighted sum along z with trapezoidal integration weights

        for j in range(self._nz):
            self._temparray = temparray * self._green_func[j, :, :, :]
            self._temparray *= self._mu
            self._temparray1[j, :, :] = self._temparray.sum(axis=0)

        # Compute Inverse Fourier transform along first 2 axis
        temparray = np.fft.fftshift(np.fft.ifftn(self._temparray1, axes=(1, 2)), axes=(1, 2))

        # Copy into output appropriately
        output += temparray[:, self._start_index:(self._end_index + 1), self._start_index:(self._end_index + 1)]

    # @staticmethod
    # def numba_accelerated_green_func_calc(green_func, num_bins, nz, m, kx, ky, z, r, k):
    #
    #     j = np.complex(0, 1)
    #     nu = j * np.sqrt(k - 0.25)
    #     cutoff = np.sqrt(2.0)
    #
    #     r1 = (cutoff / m) * r
    #
    #     for j1 in range(nz):
    #         for j2 in range(nz):
    #
    #             if j1 != j2:
    #                 f1 = z[j1] * z[j2]
    #                 f2 = f1 ** 0.5
    #                 utilde = 1.0 + (0.5 / f1) * (r1 ** 2.0 + (z[j1] - z[j2]) ** 2.0)
    #                 utildesq = utilde ** 2.0
    #                 f1 = (utildesq - 1.0) ** 0.5
    #                 galpha = ((utilde + f1) ** (-1.0 * nu)) / (f1 * f2)
    #
    #                 for i1 in range(num_bins):
    #                     for i2 in range(num_bins):
    #                         green_func[j1, j2, i1, i2] = (1.0 / (m * cutoff)) * np.sum(r1 * bessel0 * galpha)

    def __calculate_green_func(self):
        t1 = time.time()
        print("Starting Green's Functioncalculation ")

        kx, ky = np.meshgrid(self._kgrid, self._kgrid, indexing="ij")
        kabs = (kx ** 2 + ky ** 2) * 0.5
        r = np.linspace(start=0.0, stop=1.0, num=self._m, endpoint=False)
        # self.numba_accelerated_green_func_calc(
        #     green_func=self._green_func,
        #     num_bins=self._num_bins,
        #     nz=self._nz,
        #     m=self._m,
        #     kx=kx,
        #     ky=ky,
        #     z=self._zgrid,
        #     r=r,
        #     k=self._k
        # )
        self._green_func = np.fft.fftshift(self._green_func, axes=(2, 3))

        t2 = time.time()
        print("Computing 3d Green's Function took ", "{:6.2f}".format(t2 - t1), " s\n")

    def __initialize_class(self):
        # Calculate number of grid points for the domain [-2, 2] along one horizontal axis,
        # and index to crop the physical domain [-0.5, 0.5]
        self._num_bins = 4 * (self._n - 1)
        self._start_index = 3 * int((self._n - 1) / 2)
        self._end_index = 5 * int((self._n - 1) / 2)

        # Calculate horizontal grid spacing d
        # Calculate horizontal grid of wavenumbers for any 1 dimension
        self._d = 1.0 / (self._n - 1)
        self._kgrid = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n=self._num_bins, d=self._d))

        # Calculate z grid spacing d
        # Calculate z grid coordinates
        self._dz = (self._b - self._a) / (self._nz - 1)
        self._zgrid = np.linspace(start=self._a, stop=self._b, num=self._nz, endpoint=True)

        # Calculate FT of Truncated Green's Function and apply fftshift
        self._green_func = np.zeros(shape=(self._nz, self._nz, self._num_bins, self._num_bins), dtype=self._precision)
        self.__calculate_green_func()

        # Calculate integration weights
        self._mu = np.zeros(shape=(self._nz, 1, 1), dtype=np.float64) + 1.0
        self._mu[0, 0, 0] = 0.5
        self._mu[self._nz - 1, 0, 0] = 0.5

        # Allocate temporary array to avoid some reallocation
        self._temparray = np.zeros(shape=(self._nz, self._num_bins, self._num_bins), dtype=self._precision)
        self._temparray1 = np.zeros(shape=(self._nz, self._num_bins, self._num_bins), dtype=self._precision)


if __name__ == "__main__":

    n_ = 101
    nz_ = 101
    k_ = 50.0
    a_ = 0.8
    b_ = 1.8
    m_ = 100
    precision_ = np.complex64

    op = TruncatedKernelLinearIncreasingVel3d(
        n=n_,
        nz=nz_,
        k=k_,
        a=a_,
        b=b_,
        m=m_,
        precision=precision_
    )
