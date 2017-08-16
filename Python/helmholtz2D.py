import math
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt

"""
Set up global parameters, constants and simulation box parameters
"""
# Define global constants
i = 1j
pi = math.pi

# Define box extents and grid along X and Y dimensions, PML parameters
startX = 0.
endX = 1.
etaX = 0.1
nX = 51
startY = 0.
endY = 1.
etaY = 0.1
nY = 51

C = 20.0

# Calculate number of cells and grid spacing along X and Y
sizeX = (endX - startX)
sizeY = (endY - startY)
ncellsX = nX + 1
hX = sizeX / ncellsX
ncellsY = nY + 1
hY = sizeY / ncellsY

# Calculate start and end of PML boundary box along X and Y
pmlX = etaX * sizeX
startbcX = startX + pmlX
endbcX = endX - pmlX
pmlY = etaY * sizeY
startbcY = startY + pmlY
endbcY = endY - pmlY

# Setup frequency parameters
f = 5.0
omega = 2 * pi * f

"""
Initialize the matrix to zeros
"""
def initialize_matrix():
    N = nX * nY
    return np.zeros((N, N), dtype = np.complex64)

"""
Initialize the velocities to a constant c
"""
def initialize_constant_velocities(c):
    vel = np.zeros(nX * nY, dtype = np.complex64)
    vel += c
    return vel

"""
Initialize the source to zeros
"""
def initialize_source():
    src = np.zeros(nX * nY, dtype = np.complex64)
    return src

"""
Calculate sigma value along X at value x
"""
def sigmaX_val(x):
    if (x >= startbcX) and (x <= endbcX):
        return 0
    elif (x < startbcX) and (x >= startX):
        return (C / pmlX) * ((x - startbcX) / pmlX)**2
    elif (x > endbcX) and (x <= endX):
        return (C / pmlX) * ((x - endbcX) / pmlX)**2
    elif (x < startX) or (x > endX):
        return None

"""
Calculate sigma value along Y at value y
"""
def sigmaY_val(y):
    if (y >= startbcY) and (y <= endbcY):
        return 0
    elif (y < startbcY) and (y >= startY):
        return (C / pmlY) * ((y - startbcY) / pmlY)**2
    elif (y > endbcY) and (y <= endY):
        return (C / pmlY) * ((y - endbcY) / pmlY)**2
    elif (y < startY) or (y > endY):
        return None

"""
Calculate s value along X at value x
"""
def sX_val(x):
    sigma = sigmaX_val(x)
    if sigma == None:
        return None
    else:
        r = sigma / omega
        return 1.0 / (1 + r * i)
        
"""
Calculate s value along Y at value y
"""
def sY_val(y):
    sigma = sigmaY_val(y)
    if sigma == None:
        return None
    else:
        r = sigma / omega
        return 1.0 / (1 + r * i)

"""
Fill the matrix
Fast axis is along X dimension
"""
def fill_Helmholtz_matrix(A, vel):
    ###########################################################################
    # Interior nodes
    for iy in range(2, nY):
        n1 = (iy - 1) * nX
        p1Y = sY_val(startY + iy * hY) / hY
        p2Y = sY_val(startY + (iy + 0.5) * hY) / hY
        p3Y = sY_val(startY + (iy - 0.5) * hY) / hY
        
        for ix in range(2, nX):
            n2 = n1 + ix - 1
            p1X = sX_val(startX + ix * hX) / hX
            p2X = sX_val(startX + (ix + 0.5) * hX) / hX
            p3X = sX_val(startX + (ix - 0.5) * hX) / hX
            
            A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
            A[n2][n2 + 1] = p1X * p2X
            A[n2][n2 - 1] = p1X * p3X
            A[n2][n2 + nX] = p1Y * p2Y
            A[n2][n2 - nX] = p1Y * p3Y
            
    ###########################################################################
    # Edges except corners
    
    # 1. Bottom
    n1 = 0
    p1Y = sY_val(startY + hY) / hY
    p2Y = sY_val(startY + 1.5 * hY) / hY
    p3Y = sY_val(startY + 0.5 * hY) / hY
    
    for ix in range(2, nX):
        n2 = n1 + ix - 1
        p1X = sX_val(startX + ix * hX) / hX
        p2X = sX_val(startX + (ix + 0.5) * hX) / hX
        p3X = sX_val(startX + (ix - 0.5) * hX) / hX
        
        A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
        A[n2][n2 + 1] = p1X * p2X
        A[n2][n2 - 1] = p1X * p3X
        A[n2][n2 + nX] = p1Y * p2Y
    
    # 2. Top
    n1 = (nY - 1) * nX
    p1Y = sY_val(startY + nY * hY) / hY
    p2Y = sY_val(startY + (nY + 0.5) * hY) / hY
    p3Y = sY_val(startY + (nY - 0.5) * hY) / hY
    
    for ix in range(2, nX):
        n2 = n1 + ix - 1
        p1X = sX_val(startX + ix * hX) / hX
        p2X = sX_val(startX + (ix + 0.5) * hX) / hX
        p3X = sX_val(startX + (ix - 0.5) * hX) / hX
        
        A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
        A[n2][n2 + 1] = p1X * p2X
        A[n2][n2 - 1] = p1X * p3X
        A[n2][n2 - nX] = p1Y * p3Y
    
    # 3. Left
    n1 = 0
    p1X = sX_val(startX + hX) / hX
    p2X = sX_val(startX + 1.5 * hX) / hX
    p3X = sX_val(startX + 0.5 * hX) / hX
    for iy in range(2, nY):
        n2 = n1 + (iy - 1) * nX
        p1Y = sY_val(startY + iy * hY) / hY
        p2Y = sY_val(startY + (iy + 0.5) * hY) / hY
        p3Y = sY_val(startY + (iy - 0.5) * hY) / hY
            
        A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
        A[n2][n2 + 1] = p1X * p2X
        A[n2][n2 + nX] = p1Y * p2Y
        A[n2][n2 - nX] = p1Y * p3Y
    
    # 4. Right
    n1 = nX - 1
    p1X = sX_val(startX + nX * hX) / hX
    p2X = sX_val(startX + (nX + 0.5) * hX) / hX
    p3X = sX_val(startX + (nX - 0.5) * hX) / hX
    for iy in range(2, nY):
        n2 = n1 + (iy - 1) * nX
        p1Y = sY_val(startY + iy * hY) / hY
        p2Y = sY_val(startY + (iy + 0.5) * hY) / hY
        p3Y = sY_val(startY + (iy - 0.5) * hY) / hY
            
        A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
        A[n2][n2 - 1] = p1X * p3X
        A[n2][n2 + nX] = p1Y * p2Y
        A[n2][n2 - nX] = p1Y * p3Y

    ###########################################################################
    # Corners
    
    # 1. Bottom Left
    n2 = 0
    p1Y = sY_val(startY + hY) / hY
    p2Y = sY_val(startY + 1.5 * hY) / hY
    p3Y = sY_val(startY + 0.5 * hY) / hY
    p1X = sX_val(startX + hX) / hX
    p2X = sX_val(startX + 1.5 * hX) / hX
    p3X = sX_val(startX + 0.5 * hX) / hX
    
    A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
    A[n2][n2 + 1] = p1X * p2X
    A[n2][n2 + nX] = p1Y * p2Y
    
    # 2. Bottom Right
    n2 = nX - 1
    p1Y = sY_val(startY + hY) / hY
    p2Y = sY_val(startY + 1.5 * hY) / hY
    p3Y = sY_val(startY + 0.5 * hY) / hY
    p1X = sX_val(startX + nX * hX) / hX
    p2X = sX_val(startX + (nX + 0.5) * hX) / hX
    p3X = sX_val(startX + (nX - 0.5) * hX) / hX
    
    A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
    A[n2][n2 - 1] = p1X * p3X
    A[n2][n2 + nX] = p1Y * p2Y

    # 3. Top Left
    n2 = (nY - 1) * nX
    p1Y = sY_val(startY + nY * hY) / hY
    p2Y = sY_val(startY + (nY + 0.5) * hY) / hY
    p3Y = sY_val(startY + (nY - 0.5) * hY) / hY
    p1X = sX_val(startX + hX) / hX
    p2X = sX_val(startX + 1.5 * hX) / hX
    p3X = sX_val(startX + 0.5 * hX) / hX

    A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
    A[n2][n2 + 1] = p1X * p2X
    A[n2][n2 - nX] = p1Y * p3Y
    
    # 4. Top Right
    n2 = (nY - 1) * nX + nX - 1
    p1Y = sY_val(startY + nY * hY) / hY
    p2Y = sY_val(startY + (nY + 0.5) * hY) / hY
    p3Y = sY_val(startY + (nY - 0.5) * hY) / hY
    p1X = sX_val(startX + nX * hX) / hX
    p2X = sX_val(startX + (nX + 0.5) * hX) / hX
    p3X = sX_val(startX + (nX - 0.5) * hX) / hX

    A[n2][n2] = - p1X * (p3X + p2X) - p1Y * (p3Y + p2Y)
    A[n2][n2 - 1] = p1X * p3X
    A[n2][n2 - nX] = p1Y * p3Y

    ###########################################################################
    # Add diagonal terms
    for iy in range(nY):
        n1 = iy * nX
        for ix in range(nX):
            n2 = n1 + ix
            A[n2][n2] = A[n2][n2] + (omega / vel[n2]) ** 2
    
    
    return A
   
"""
Fill the source with an impulse
"""
def fill_source_impulse(src, amp, ix, iy):
    src[nX * iy + ix] = amp
    return src

"""
Fill the source with a dipole
"""
def fill_source_dipole(src, amp, ix, iy):
    src[nX * iy + ix] = amp
    src[nX * iy + ix + 1] = -amp
    return src

"""
Fill the source with a small line extended source
"""
def fill_source_line(src, amp, ix1, ix2, iy):
    for i1 in range(ix1, ix2 + 1):
        src[nX * iy + i1] = amp
    return src

"""
Solve for wavefield
"""
def solve(A, src):
    u = np.linalg.solve(A,src)
    return u

"""
Plot the sparsity pattern
"""
def plot_sparsity(A):
    plt.figure()
    plt.title('Sparsity pattern (A)')
    plt.xlabel('Cols')
    plt.ylabel('Rows')
    plt.imshow(np.abs(A), interpolation='none', cmap='Greys')
    plt.colorbar()

"""
Plot the eigenvalues
"""
def plot_eigenvalues(eigvals):
    evalReal = np.real(eigvals)
    evalImag = np.imag(eigvals)
    l = max(max(abs(evalReal)), max(abs(evalImag)))
    plt.figure()
    plt.title('Eigenvalues (A)')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.xlim([-l,l])
    plt.ylim([-l,l])
    plt.grid(True)
    plt.scatter(evalReal, evalImag)

"""
Plot the solution
"""
def plot(u):
    uReal = np.reshape(np.real(u), (nX, nY))
    uImag = np.reshape(np.imag(u), (nX, nY))
    uAbs = np.reshape(np.abs(u), (nX, nY))
    
    plt.figure()
    plt.subplot(121)
    plt.title('Real(u)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(color='w')
    plt.imshow(uReal)
    plt.colorbar()
    
    plt.subplot(122)
    plt.title('Imag(u)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(color='w')
    plt.imshow(uImag)
    plt.colorbar()
    plt.show()

"""
Main program starts here
"""
if __name__ == "__main__":
    
    A = initialize_matrix()
    vel = initialize_constant_velocities(1.0)
    src = initialize_source()
    A = fill_Helmholtz_matrix(A, vel)
    
    #plot_sparsity(A)
    
    #evals = np.linalg.eigvals(A)
    #plot_eigenvalues(evals)
    
    src = fill_source_impulse(src, 1e4, nX / 2, nY / 2)
    u = solve(A,src)
    plot(u)
    
    #src = fill_source_dipole(src, 1e4, nX / 2, nY / 2)
    #u = solve(A,src)
    #plot(u)
    
    #src = fill_source_line(src, 1e4, nX / 2 - 20, nX / 2 + 20, nY / 2)
    #u = solve(A,src)
    #plot(u)