# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:11:15 2019

@author: rahul
"""
from Utilities import TypeChecker
import math
import numpy as np


def ricker_time(freq_peak=10.0, nt=250, dt=0.004, delay=0.05):

    TypeChecker.check_float_positive(x=freq_peak)
    TypeChecker.check_int_positive(x=nt)
    TypeChecker.check_float_positive(x=dt)
    TypeChecker.check(x=delay, expected_type=(float, int))

    t = np.arange(0.0, nt * dt, dt)
    y = (1.0 - 2.0 * (math.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2)) \
        * np.exp(-(math.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2))
    return t, y


def ricker_wavelet_coefficient(omega, omega_peak, scale=1e6):

    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=omega_peak)

    amp = 2.0 * (omega ** 2.0) / (np.sqrt(math.pi) * (omega_peak ** 3.0)) * np.exp(-(omega / omega_peak) ** 2)
    return scale * np.complex64(amp)
