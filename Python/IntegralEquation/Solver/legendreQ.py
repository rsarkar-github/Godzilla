import numpy as np
import scipy.special as sp
import numba


def legendreQ(nu, z):
    """
    Computes legendre Q function using relation to hypergeometric function
    Q_nu(z) = f * hypgeom((1+nu)/2, (2+nu)/2, nu + 3/2, 1/z^2)
    where, f = sqrt(pi) * gamma(nu + 1) / (2^ (nu+1) * gamma(nu + 3/2)), and |z| > 1

    :param nu: complex, but not a negative integer
    :param z: complex, |z| < 1
    :return: Q_nu(z)
    """

    




if __name__ == "__main__":

    nu = 2 + complex(0, 1)
    z = 0.35
    x = legendreQ(nu, z)
    print("x = ", x)
