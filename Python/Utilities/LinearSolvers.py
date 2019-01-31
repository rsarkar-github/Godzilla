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

    TypeChecker.check_float_positive(x=niter)

    # Calculate initial residual, and residual norm
    r = rhs - linear_operator(x0)
    r_norm = np.linalg.norm(x=r)
    if r_norm < 1e-10:
        return x0
    r_norm_sq = r_norm ** 2

    # Initialize p
    p = copy.deepcopy(r)

    # Initialize residual array, iteration array
    residual = [r_norm]
    iterations = [0]

    # Run CG iterations
    for num_iter in range(niter):

        # Compute A*p and alpha
        matrix_times_p = linear_operator(p)
        alpha = r_norm_sq / np.vdot(p, matrix_times_p)

        # Update x0, residual
        x0 += alpha * p
        r -= alpha * matrix_times_p

        # Calculate beta
        r_norm_new = np.linalg.norm(x=r)
        r_norm_new_sq = r_norm_new ** 2
        beta = r_norm_new_sq / r_norm_sq

        # Check convergence
        if r_norm_new < 1e-10:
            break

        # Update p, residual norm
        p = r + beta * p
        r_norm_sq = r_norm_new_sq

        # Update residual array, iteration array
        residual.append(r_norm_new)
        iterations.append(num_iter)

    return x0, (iterations, residual)
