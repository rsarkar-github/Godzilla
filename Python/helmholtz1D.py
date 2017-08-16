# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:16:56 2017

@author: rahul
"""
import math
import numpy as np
import matplotlib.pyplot as plt

i = 1j
pi = math.pi
startX = 0
endX = 1
etaX = 0.1
nX = 1000
ncellsX = nX + 1
hX = 1.0 / ncellsX
startBC = startX + etaX
endBC = endX - etaX
f = 10.0
omega = 2 * pi * f
c = 1.0

C = 10.0

def sigma_val(x):
    if (x >= startBC) and (x <= endBC):
        return 0
    elif (x < startBC) and (x >= startX):
        return (C / etaX) * ((x - startBC) / etaX)**2
    elif (x > endBC) and (x <= endX):
        return (C / etaX) * ((x - endBC) / etaX)**2
    elif (x < startX) or (x > endX):
        return None

def s_val(x):
    sigma = sigma_val(x)
    if sigma == None:
        return None
    else:
        r = sigma / omega
        return 1.0 / (1 + r * i)

# Create a dense matrix for testing
A = np.zeros((nX,nX), dtype = np.complex64)
b = np.zeros((nX,1), dtype = np.complex64)

# interior nodes
for k in range(2,nX):
    
    p1 = s_val(startX + k * hX) / hX
    p2 = s_val(startX + (k + 0.5) * hX) / hX
    p3 = s_val(startX + (k - 0.5) * hX) / hX
    
    A[k-1][k-2] = p1 * p3
    A[k-1][k] = p1 * p2
    A[k-1][k-1] = (omega / c)**2 - (A[k-1][k-2] + A[k-1][k])
    
# boundary nodes
p1 = s_val(startX + hX) / hX
p2 = s_val(startX + 1.5 * hX) / hX
p3 = s_val(startX + 0.5 * hX) / hX
A[0][1] = p1 * p2
A[0][0] = (omega / c)**2 - A[0][1] - p1 * p3

p1 = s_val(endX - hX) / hX
p2 = s_val(endX - 0.5 * hX) / hX
p3 = s_val(endX - 1.5 * hX) / hX
A[nX-1][nX-2] = p1 * p3
A[nX-1][nX-1] = (omega / c)**2 - A[nX-1][nX-2] - p1 * p2

# Fill b
b[nX/2][0] = 1000.0

# Solve
x = np.linalg.solve(A,b)

# Plot
plt.plot(np.real(x))
plt.plot(np.imag(x))
plt.plot(np.abs(x))