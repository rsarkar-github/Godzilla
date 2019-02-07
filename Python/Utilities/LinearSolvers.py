# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:11:15 2019

@author: rahul
"""
from Utilities import TypeChecker
import copy
import numpy as np


"""
TODO:
1. Add exception handling to the methods
"""


def conjugate_gradients(linear_operator, rhs, x0, niter):

    TypeChecker.check_int_positive(x=niter)

    # Get rhs norm
    fac = np.linalg.norm(x=rhs)
    print("Norm of rhs = ", fac)
    if fac < 1e-15:
        raise ValueError("Norm of rhs < 1e-15. Trivial solution of zero. Scale up the problem.")

    # Scale rhs
    rhs_new = rhs / fac
    x = x0 / fac

    # Calculate initial residual, and residual norm
    r = rhs_new - linear_operator(x)
    r_norm = np.linalg.norm(x=r)
    if r_norm < 1e-12:
        return x0, [r_norm]
    r_norm_sq = r_norm ** 2

    # Initialize p
    p = copy.deepcopy(r)

    # Initialize residual array, iteration array
    residual = [r_norm]
    objective = [np.real(0.5 * np.vdot(x, linear_operator(x)) - np.vdot(x, rhs_new))]

    # Run CG iterations
    for num_iter in range(niter):

        print(
            "Beginning iteration : ", num_iter,
            " , Residual : ", residual[num_iter],
            " , Objective : ", objective[num_iter]
        )

        # Compute A*p and alpha
        matrix_times_p = linear_operator(p)
        alpha = r_norm_sq / np.vdot(p, matrix_times_p)

        # Update x0, residual
        x += alpha * p
        r -= alpha * matrix_times_p

        # Calculate beta
        r_norm_new = np.linalg.norm(x=r)
        r_norm_new_sq = r_norm_new ** 2
        beta = r_norm_new_sq / r_norm_sq

        # Check convergence
        if r_norm_new < 1e-12:
            break

        # Update p, residual norm
        p = r + beta * p
        r_norm_sq = r_norm_new_sq

        # Update residual array, iteration array
        residual.append(r_norm_new)
        objective.append(np.real(0.5 * np.vdot(x, linear_operator(x)) - np.vdot(x, rhs_new)))

    # Remove the effect of the scaling
    x = x * fac

    return x, residual


if __name__ == "__main__":

    # Create a trial problem
    dim = 30

    a = np.zeros((dim, dim), dtype=np.complex64)
    a += np.random.rand(dim, dim).astype(dtype=np.complex64) \
         + 1j * np.random.rand(dim, dim).astype(dtype=np.complex64)
    q, _ = np.linalg.qr(a)
    d = np.random.uniform(low=1, high=2, size=(dim,)).astype(dtype=np.complex64)

    mat = np.dot(np.diag(d), q)
    mat = np.dot(np.conjugate(np.transpose(q)), mat)
    vector_true = np.random.rand(dim).astype(dtype=np.complex64) \
                  + 1j * np.random.rand(dim).astype(dtype=np.complex64)
    b = np.dot(mat, vector_true)
    x0 = np.zeros((dim,), dtype=np.complex64)

    # The operator
    def linop(x):
        return np.dot(mat, x)

    # Solve
    x0, res = conjugate_gradients(linear_operator=linop, rhs=b, x0=x0, niter=10)

    # Print solution
    print("\nTrue solution:")
    print(vector_true)

    print("\nCG solution:")
    print(x0)

    print("\nDifference:")
    print(vector_true - x0)

    print("\nResiduals:")
    print(res)