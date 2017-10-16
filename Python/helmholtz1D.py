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
etaX = 0.15
nX = 500
ncellsX = nX + 1
hX = 1.0 / ncellsX
startBC = startX + etaX
endBC = endX - etaX
f = 50.0
omega = 2 * pi * f
c = 1.0

C = 100.0

# Define time parameters and wavelet parameters
nT = 200
dt = 0.008
peak_freq_ricker = 10
delay = 0.2

# Define spatial Gaussian parameters
mean = 0.5
sigma = 0.01
amp = 1000.0

# Define sampling point
sampling_point = int(3 * nX / 4)

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

def create_gaussian_source(b, mean, sigma, amp):
    for k in range(nX):
        b[k][0] = np.exp(-0.5 * (((k+1) * hX - mean)/sigma)**2) + 0 * i
    b = b / np.sqrt(2 * pi * sigma * sigma)
    b = b * amp
    return b;
    
def create_ricker(peak_freq, dt, nT, delay):
    val = np.zeros((nT,1), dtype = np.complex64)
    for k in range(nT):
        temp = (pi * peak_freq * (k * dt - delay))**2
        val[k][0] = (1 - 2 * temp) * np.exp(-temp)
    return val
    
def create_freq_list(nT, dt):
    freq_list = np.zeros((nT,1), dtype = np.complex64)
    dfreq = 1.0 / (nT * dt)
    for k in range(nT/2 + 1):
        freq_list[k][0] = k * dfreq
    for k in range(1, nT / 2):
        freq_list[nT - k][0] = -k * dfreq
    return freq_list

def create_A_matrix(A, omega):
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
    
    return A

def calculate_Laplacian(lap, x, hX):
    n = int(x.shape[0])
    lap[0][0] = x[1][0] - 2 * x[0][0]
    lap[n - 1][0] = x[n - 2][0] - 2 * x[n - 1][0]
    for k in range(1,n-1):
        lap[k][0] = x[k - 1][0] - 2 * x[k][0] + x[k + 1][0]
    lap /= hX**2

def apply_taper(x, hX):
    nX = int(x.shape[0])
    for k in range(nX):
        coord = (k + 1) * hX
        if(coord < etaX):
            x[k][0] *= np.exp(-0.03 * (1 - coord / etaX))
        if(coord > (endX - etaX)):
            x[k][0] *= np.exp(-0.03 * (1 + (coord - endX) / etaX))
    
def time_domain_solution(wavelet, spatial_weight, dt, hX, sampling_point):
    # Get nT, nX
    nT = int(wavelet.shape[0])
    nX = int(spatial_weight.shape[0])
    
    # Create placeholder for output
    wave_sampled = np.zeros((nT,1), dtype = np.float64)
    
    # Create fields for temporary wavefields
    up1 = np.zeros((nX,1), dtype = np.float64)
    up2 = np.zeros((nX,1), dtype = np.float64)
    u = np.zeros((nX,1), dtype = np.float64)
    
    # Remaining time steps
    for k in range(1,nT):
        print "k = ", k
        calculate_Laplacian(u, up1, hX)
        u *= (c * dt)**2
        u += 2 * up1 - up2 + wavelet[k - 1] * ((c * dt)**2) * spatial_weight
        apply_taper(u, hX)
        apply_taper(up1, hX)
        up2 = 1 * up1
        up1 = 1 * u
        wave_sampled[k][0] = u[sampling_point][0]    
    return wave_sampled

def resample_time(sol, resamp_factor):
    nT = int(sol.shape[0])
    sol_resamp = np.zeros((nT / resamp_factor,1), dtype = np.float64)
    for k in range(nT / resamp_factor):
        sol_resamp[k][0] = sol[k * resamp_factor][0]
    return sol_resamp

# Create Ricker wavelet and Fourier transform it
wavelet = create_ricker(peak_freq_ricker, dt, nT, delay)
wavelet_fft = -np.fft.ifft(wavelet, axis=0)

# Create and fill b
b = np.zeros((nX,1), dtype = np.complex64)
b = create_gaussian_source(b, mean, sigma, amp)

# Create Ricker wavelet for time propagation
resamp_factor = 10
wavelet_time = np.real(create_ricker(peak_freq_ricker, dt/resamp_factor, nT*resamp_factor, delay))

# Time propagation
sol_timeprop = time_domain_solution(wavelet_time, np.real(b), dt/resamp_factor, hX, sampling_point)
sol_timeprop = resample_time(sol_timeprop, resamp_factor)

# FFT the timeprop solution
sol_timeprop_fft = np.fft.ifft(sol_timeprop, axis=0)

# Create a dense matrix for testing
A = np.zeros((nX,nX), dtype = np.complex64)

# Create placeholder for solution at sample point
sol_fft = np.zeros((nT,1), dtype = np.complex64)


# Create list of frequencies
freq_list =  create_freq_list(nT, dt)

# Solve Helmholtz equation in a loop for half of the frequencies (except 0)
# For remaining use symmetry
rhs = b
nStart = 1
nEnd = 50
for k in range(nStart, nEnd):
    # Print a line
    print "k = ", k
    
    # Update matrix A
    A = create_A_matrix(A, 2 * pi * freq_list[k][0])
    
    # Update rhs by scaling b by freq
    rhs = b * wavelet_fft[k][0]
    
    # Solve for x
    x = np.linalg.solve(A, rhs)
    
    # Extract solution
    sol_fft[k][0] = x[sampling_point][0]
    sol_fft[nT - k][0] = np.conj(x[sampling_point][0])

# Inverse fft the solution at sampled point
sol = np.fft.fft(sol_fft, axis=0)

#plt.figure(1)
#plt.plot(sol,'-r*')
#plt.plot(sol_timeprop,'-k*')

plt.figure(1)
plt.plot(np.real(sol_fft),'-r*')
plt.plot(np.real(sol_timeprop_fft),'-k*')

plt.figure(2)
plt.plot(np.imag(sol_fft),'-r*')
plt.plot(np.imag(sol_timeprop_fft),'-k*')

plt.figure(3)
plt.plot(np.abs(sol_fft),'-r*')
plt.plot(np.abs(sol_timeprop_fft),'-k*')

## Plot
#plt.plot(np.real(x),'r')
#plt.plot(np.imag(x),'b')
#plt.plot(np.abs(x),'g')