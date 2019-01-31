# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:04:30 2017
@author: rahul
"""

import math
import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt

class Common(object):

    """
    This class defines the set of global parameters for the problem
    """
    ppw = 10                        # Points per wavelength
    pml = 1.0                       # Width of pml layer in units of wavelength
    pml_damping = 50                # Default pml damping parameter
    dim = 1.0                       # Length of simulation box in km

    omega = 100.0                   # Default angular frequency in rad/s

    i = 1j                          # Complex number i
    pi = math.pi                    # Math constant pi

    vel = 1.0                       # Default velocity in km/s
