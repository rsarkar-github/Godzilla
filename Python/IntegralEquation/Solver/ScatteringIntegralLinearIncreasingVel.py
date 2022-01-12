import numpy as np
import numba
import scipy.special as sp
import time
from . import TypeChecker
from . import SpecialFunc
import matplotlib.pyplot as plt


class TruncatedKernelLinearIncreasingVel3d:

    def __init__(self, n, nz, k, a, b, m, precision):
        """
        :param n: The field will have shape n x n x nz, with n odd. This means that the cube
        [-0.5, 0.5]^2 is gridded into n^2 points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
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

        self._cutoff = np.sqrt(2.0)

        # Run class initializer
        self.__initialize_class()

    def apply_kernel(self, u, output):
        """
        :param u: 3d numpy array (must be nz x n x n dimensions with n odd).
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

        # Restore class temporary arrays
        self._temparray *= 0
        self._temparray1 *= 0

    @property
    def greens_func(self):
        return self._green_func

    @staticmethod
    @numba.jit(nopython=True)
    def green_func_calc(green_func, nz, m, z, r, bessel0, k):

        j = complex(0, 1)
        nu = j * np.sqrt(k**2 - 0.25)
        cutoff = np.sqrt(2.0)
        r_scaled = cutoff * r

        print("Total z slices = ", nz, "\n")
        for j1 in range(nz):

            print("Slice number = ", j1 + 1)
            for j2 in range(j1, nz):

                if j1 != j2:
                    f1 = z[j1] * z[j2]
                    utilde = 1.0 + (0.5 / f1) * (r_scaled ** 2.0 + (z[j1] - z[j2]) ** 2.0)
                    utildesq = utilde ** 2.0
                    f2 = (utildesq - 1.0) ** 0.5
                    galpha = ((utilde + f2) ** (-1.0 * nu)) / (f2 * (f1 ** 0.5))
                    green_func[j1, j2, :, :] = (1.0 / (m * cutoff)) * np.sum((r_scaled * galpha) * bessel0, axis=0)
                    green_func[j2, j1, :, :] = green_func[j1, j2, :, :]

                else:
                    f1 = z[j1]
                    utilde = (2.0 / (f1 ** 2) + (r ** 2) / (f1 ** 4)) ** 0.5
                    galpha_times_r_num = (1 + (r ** 2) / (f1 ** 2) + r * utilde) ** (-1 * nu)
                    galpha_times_r = galpha_times_r_num / (f1 * utilde)
                    green_func[j1, j2, :, :] = np.sum(galpha_times_r * bessel0, axis=0) / m
                    green_func[j2, j1, :, :] = green_func[j1, j2, :, :]

    def __calculate_green_func(self):
        t1 = time.time()
        print("\nStarting Green's Function calculation ")

        kx, ky = np.meshgrid(self._kgrid, self._kgrid, indexing="ij")
        kabs = (kx ** 2 + ky ** 2) ** 0.5
        r = np.reshape(np.linspace(start=0.0, stop=1.0, num=self._m, endpoint=False), newshape=(self._m, 1, 1))
        r_scaled = self._cutoff * r
        bessel0 = sp.j0(r_scaled * kabs)

        self.green_func_calc(
            green_func=self._green_func,
            nz=self._nz,
            m=self._m,
            z=self._zgrid,
            r=r,
            bessel0=bessel0,
            k=self._k
        )
        self._green_func = np.fft.fftshift(self._green_func, axes=(2, 3))

        t2 = time.time()
        print("\nComputing 3d Green's Function took ", "{:6.2f}".format(t2 - t1), " s\n")

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


class TruncatedKernelLinearIncreasingVel2d:

    def __init__(self, n, nz, k, a, b, m, precision):
        """
        :param n: The field will have shape n x nz, with n odd. This means that the unit cube
        [-0.5, 0.5] is gridded into n points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x nz, with n odd. This means that the interval [a, b] is gridded
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

        self._cutoff = np.sqrt(1.0)

        # Run class initializer
        self.__initialize_class()

    def apply_kernel(self, u, output, adj=False, add=False):
        """
        :param u: 2d numpy array (must be nz x n dimensions with n odd).
        :param output: 2d numpy array (same dimension as u). Assumed to be zeros.
        :param adj: Boolean flag (forward or adjoint operator)
        :param add: Boolean flag (whether to add result to output)
        """

        # Check types of input
        if u.dtype != self._precision or output.dtype != self._precision:
            raise TypeError("Types of 'u' and 'output' must match that of class: ", self._precision)

        # Check dimensions
        if u.shape != (self._nz, self._n) or output.shape != (self._nz, self._n):
            raise ValueError("Shapes of 'u' and 'output' must be (nz, n) with nz = ", self._nz, " and n = ", self._n)

        # Copy u into self._temparray
        self._temparray[:, self._start_index:(self._end_index + 1)] = u

        # Compute Fourier transform along first axis
        temparray = np.fft.fftn(np.fft.fftshift(self._temparray, axes=(1,)), axes=(1,))

        # Forward mode
        # Do the following for each z slice
        # 1. Multiply with Fourier transform of Truncated Kernel Green's function slice for that z
        # 2. Compute weighted sum along z with trapezoidal integration weights
        if not adj:
            for j in range(self._nz):
                self._temparray = temparray * self._green_func[j, :, :]
                self._temparray *= self._mu
                self._temparray1[j, :] = self._temparray.sum(axis=0)

            # Compute Inverse Fourier transform along first axis
            temparray = np.fft.fftshift(np.fft.ifftn(self._temparray1, axes=(1,)), axes=(1,))

            # Copy into output appropriately
            if not add:
                output *= 0
            output += temparray[:, self._start_index:(self._end_index + 1)]

            # Restore class temporary arrays
            self._temparray *= 0
            self._temparray1 *= 0

        # Adjoint mode
        # Do the following for each z slice
        # 1. Multiply with Fourier transform of Truncated Kernel Green's function slice for that z (complex conjugate)
        # 2. Compute sum along z
        # 3. Multiply with integration weight for the z slice
        if adj:
            for j in range(self._nz):
                self._temparray = temparray * self._green_func_conj[j, :, :]
                self._temparray1[j, :] = self._mu[j, 0] * self._temparray.sum(axis=0)

            # Compute Inverse Fourier transform along first axis
            temparray = np.fft.fftshift(np.fft.ifftn(self._temparray1, axes=(1,)), axes=(1,))

            # Copy into output appropriately
            if not add:
                output *= 0
            output += temparray[:, self._start_index:(self._end_index + 1)]

            # Restore class temporary arrays
            self._temparray *= 0
            self._temparray1 *= 0

    @property
    def greens_func(self):
        return self._green_func

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def green_func_calc(green_func, nz, m, z, r, cos, k):

        j = complex(0, 1)
        nu = j * np.sqrt(k**2 - 0.25)
        lamb = nu - 0.5

        print("Total z slices = ", nz, "\n")
        for j1 in numba.prange(nz):

            print("Slice number = ", j1 + 1)
            for j2 in range(j1, nz):

                f1 = z[j1] * z[j2]
                utilde = 1.0 + (0.5 / f1) * (r ** 2.0 + (z[j1] - z[j2]) ** 2.0)
                leg = SpecialFunc.legendre_q_v1(lamb, utilde, 1e-12)

                prod = cos * 1.0
                for j4 in range(m-1):
                    prod[j4, :] *= leg[j4, 0]

                green_func[j1, j2, :] = (1.0 / (np.pi * m)) * np.sum(prod, axis=0)
                green_func[j2, j1, :] = green_func[j1, j2, :]

    def __calculate_green_func(self):
        t1 = time.time()
        print("\nStarting Green's Function calculation ")

        kx = np.meshgrid(self._kgrid, indexing="ij")
        kabs = np.abs(kx)
        r = np.reshape(np.linspace(start=0.0, stop=1.0, num=self._m, endpoint=False), newshape=(self._m, 1))
        r = r[1:]
        cos = np.cos(r * kabs) # notice scaling is 1 compared to 3d case, so r_scaled is not needed here

        self.green_func_calc(
            green_func=self._green_func,
            nz=self._nz,
            m=self._m,
            z=self._zgrid,
            r=r,
            cos=cos.astype(dtype=np.complex128),
            k=self._k
        )
        self._green_func = np.fft.fftshift(self._green_func, axes=2)
        self._green_func_conj = np.conjugate(self._green_func)

        t2 = time.time()
        print("\nComputing 2d Green's Function took ", "{:6.2f}".format(t2 - t1), " s\n")

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
        self._green_func = np.zeros(shape=(self._nz, self._nz, self._num_bins), dtype=self._precision)
        self.__calculate_green_func()

        # Calculate integration weights
        self._mu = np.zeros(shape=(self._nz, 1), dtype=np.float64) + 1.0
        self._mu[0, 0] = 0.5
        self._mu[self._nz - 1, 0] = 0.5

        # Allocate temporary array to avoid some reallocation
        self._temparray = np.zeros(shape=(self._nz, self._num_bins), dtype=self._precision)
        self._temparray1 = np.zeros(shape=(self._nz, self._num_bins), dtype=self._precision)


if __name__ == "__main__":
    n_ = 101
    nz_ = 101
    k_ = 100.0
    a_ = 0.8
    b_ = 1.8
    m_ = 400
    precision_ = np.complex64

    # # 3d test
    # op = TruncatedKernelLinearIncreasingVel3d(
    #     n=n_,
    #     nz=nz_,
    #     k=k_,
    #     a=a_,
    #     b=b_,
    #     m=m_,
    #     precision=precision_
    # )
    #
    # u_ = np.zeros(shape=(nz_, n_, n_), dtype=precision_)
    # u_[int(nz_ / 8), int(n_ / 2), int(n_ / 2)] = 1.0
    # output_ = u_ * 0
    #
    # start_t_ = time.time()
    # op.apply_kernel(u=u_, output=output_)
    # end_t_ = time.time()
    # print("Total time to execute convolution: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
    #
    # scale = 1e-4
    # fig = plt.figure()
    # plt.imshow(np.real(output_[:, :, int(n_ / 2)]), cmap="Greys", vmin=-scale, vmax=scale)
    # plt.grid(True)
    # plt.title("Real")
    # plt.colorbar()
    # plt.show()
    #
    # u_ = np.zeros(shape=(nz_, n_, n_), dtype=precision_)
    # u_[int(nz_ / 2), int(n_ / 2), int(n_ / 2)] = 1.0
    # output_ = u_ * 0
    #
    # start_t_ = time.time()
    # op.apply_kernel(u=u_, output=output_)
    # end_t_ = time.time()
    # print("Total time to execute convolution: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
    #
    # scale = 1e-4
    # fig = plt.figure()
    # plt.imshow(np.real(output_[:, :, int(n_ / 2)]), cmap="Greys", vmin=-scale, vmax=scale)
    # plt.grid(True)
    # plt.title("Real")
    # plt.colorbar()
    # plt.show()
    #
    # fig.savefig(
    #     "Python/IntegralEquation/Fig/testplot.pdf",
    #     bbox_inches='tight',
    #     pad_inches=0
    # )

    # 2d test
    op = TruncatedKernelLinearIncreasingVel2d(
        n=n_,
        nz=nz_,
        k=k_,
        a=a_,
        b=b_,
        m=m_,
        precision=precision_
    )

    u_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    u_[int(nz_ / 8), int(n_ / 2)] = 1.0
    output_ = u_ * 0

    start_t_ = time.time()
    op.apply_kernel(u=u_, output=output_)
    end_t_ = time.time()
    print("Total time to execute convolution: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")

    scale = 1e-3
    fig = plt.figure()
    plt.imshow(np.real(output_), cmap="Greys", vmin=-scale, vmax=scale)
    plt.grid(True)
    plt.title("Real")
    plt.colorbar()
    plt.show()
