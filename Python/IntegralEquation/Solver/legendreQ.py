import numpy as np
from numba import vectorize, float32, float64, complex64, complex128
import time


@vectorize([complex128(complex128)], nopython=True)
def gamma(x):
    """
    Computes Gamma function using Lanczos method
    :param x: complex number (should not be a pole of gamma function)
    :return: Gamma(x)
    """
    p = [676.5203681218851,
         -1259.1392167224028,
         771.32342877765313,
         -176.61502916214059,
         12.507343278686905,
         -0.13857109526572012,
         9.9843695780195716e-6,
         1.5056327351493116e-7
         ]

    if np.real(x) < 0.5:
        # Use reflection formula
        x0 = x
        x = 1 - x0

        x -= 1
        total = 0.99999999999980993
        for (i, val) in enumerate(p):
            total += val / (x + i + 1)
        t = x + len(p) - 0.5
        y = np.sqrt(2 * np.pi) * t ** (x + 0.5) * np.exp(-t) * total
        y = np.pi / (np.sin(np.pi * x0) * y)
    else:
        x -= 1
        total = 0.99999999999980993
        for (i, val) in enumerate(p):
            total += val / (x + i + 1)
        t = x + len(p) - 0.5
        y = np.sqrt(2 * np.pi) * t ** (x + 0.5) * np.exp(-t) * total
    return y

@vectorize(
    [complex128(complex128, complex128, complex128, complex128, float64)],
    nopython=True
)
def hyp2f1(a, b, c, z, tol=1e-15):
    """
    Compute the Gauss hypergeometric function using series summation
    :param a: complex
    :param b: complex
    :param c: complex
    :param z: complex, |z| < 1
    :param tol: float, tolerance for stopping summation
    :return: hyp2f1(a,b,c,z)
    """

    # Define step (after every step iterations convergence is checked)
    step = 64

    # Compute using hypergeometric function
    alpha = np.zeros(shape=(step,), dtype=np.complex128)
    beta = np.zeros(shape=(step,), dtype=np.complex128)

    n = 0
    alpha[0] = 1
    beta[0] = 1
    for k in range(step - 1):
        alpha[k + 1] = alpha[k] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
        beta[k + 1] = beta[k] + alpha[k + 1]
        n = n + 1

    check_convergence = np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
                        np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
                        np.abs(alpha[step - 3] / beta[step - 4]) > tol

    while check_convergence:

        alpha[0] = alpha[step - 1] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
        beta[0] = beta[step - 1] + alpha[0]
        n = n + 1

        for k in range(step - 1):
            alpha[k + 1] = alpha[k] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
            beta[k + 1] = beta[k] + alpha[k + 1]
            n = n + 1

        check_convergence = np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
                            np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
                            np.abs(alpha[step - 3] / beta[step - 4]) > tol

    return beta[step - 1]

@vectorize([complex128(complex128, complex128, float64)], nopython=True)
def legendre_q(nu, z, tol=1e-15):
    """
    Computes legendre Q function using relation to hypergeometric function
    Q_nu(z) = f * hyp2f1((1+nu)/2, (2+nu)/2, nu + 3/2, 1/z^2)
    where, f = sqrt(pi) * gamma(nu + 1) / (2^ (nu+1) * gamma(nu + 3/2)), and |z| > 1

    :param nu: complex, but not a negative integer
    :param z: float, |z| > 1
    :param tol: float, tolerance for convergence checking
    :return: Q_nu(z)
    """

    # Evaluate a, b, z1 for hypergeometric function
    a = 0.5 * nu + 0.5
    b = 0.5 * nu + 1.0
    c = a + b
    z1 = z ** (-2.0)

    # Compute legendreQ function
    leg = hyp2f1(a, b, c, z1, tol) * (np.pi ** 0.5) * gamma(1.0 + nu) / (gamma(1.5 + nu) * ((2 * z) ** (1.0 + nu)))
    return leg

@vectorize([complex128(complex128, complex128, float64)], nopython=True)
def legendre_q_v1(nu, z, tol=1e-15):
    """
    Computes legendre Q function using relation to hypergeometric function
    Q_(nu - 1/2)(z) = f * hyp2f1(1/2, 1/2 + nu, 1 + nu, e^(-2 nu)),
    where f = sqrt(pi) * e^(-eta(nu + 0.5)) * gamma(nu + 1/2) / gamma(nu + 1), z = cosh(eta), and |z| > 1

    :param nu: complex, but not a negative integer
    :param z: float, |z| > 1
    :param tol: float, tolerance for convergence checking
    :return: Q_nu(z)
    """

    nu = nu + 0.5

    # Compute eta
    eta = np.arccosh(z)

    # Evaluate a, b, z1 for hypergeometric function
    a = 0.5
    b = 0.5 + nu
    c = a + b
    z1 = np.exp(-2.0 * eta)

    # Compute legendreQ function
    leg = hyp2f1(a, b, c, z1, tol) * (np.pi ** 0.5) * \
          gamma(0.5 + nu) * np.exp(-1.0 * (0.5 + nu) * eta ) / gamma(1.0 + nu)
    return leg


if __name__ == "__main__":

    # Test LegendreQ function
    num = 1000
    _nu = 1
    _z = 1.001

    _nu_arr = np.zeros(shape=(num,), dtype=np.complex128) + _nu
    _z_arr = np.zeros(shape=(num,), dtype=np.float64) + _z

    t_start = time.time()
    _x = legendre_q(_nu_arr, _z_arr, 1e-15)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (method 1) = ", _x[0])

    print("\n")
    t_start = time.time()
    _x = legendre_q_v1(_nu_arr, _z_arr, 1e-15)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (method 2) = ", _x[0])

    print("\n")
    _x1 = -1.0 + 0.5 * _z * np.log(np.abs((1.0 + _z) / (1.0 - _z)))
    print("Analytical x = ", complex(_x1))


    # # Hypergeometric function test
    # _nu_arr1 = np.zeros(shape=(num,), dtype=np.float64) + _nu
    # _a = 0.5 * _nu_arr1 + 0.5
    # _b = 0.5 * _nu_arr1 + 1.0
    # _c = _a + _b
    # _z_arr1 = _z_arr ** (-2.0)
    # t_start = time.time()
    # sp.hyp2f1(_a, _b, _c, _z_arr1)
    # t_end = time.time()
    # print("\nTime for Scipy computation (average): ", (t_end - t_start) / num, " s")
