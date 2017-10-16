# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:18:48 2017

@author: rahul
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_Laplacian(lap, x, hX):
    n = int(x.shape[0])
    lap[0][0] = x[1][0] - 2 * x[0][0]
    lap[n - 1][0] = x[n - 2][0] - 2 * x[n - 1][0]
    for k in range(1,n-1):
        lap[k][0] = x[k - 1][0] - 2 * x[k][0] + x[k + 1][0]
    lap = lap * (1 / (hX * hX))

def apply_taper(x, hX):
    nX = int(x.shape[0])
    for k in range(nX):
        coord = (k + 1) * hX
        if(coord < etaX):
            x[k][0] *= np.exp(-(1 - coord / etaX))
        if(coord > (endX - etaX)):
            x[k][0] *= np.exp(-(1 + (coord - endX) / etaX))