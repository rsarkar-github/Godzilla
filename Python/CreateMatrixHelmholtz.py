# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:04:30 2017
@author: rahul
"""
from Velocity import*
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import numpy as np
import copy
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


class CreateMatrixHelmholtz2D(object):
    """
    Create a sparse matrix object that can solve the Helmholtz equation.
    Currently the class only supports PML boundary conditions on all faces.
    """

    def __init__(self, velocity2d=Velocity2D(), pml_damping=Common.pml_damping):

        self.vel2D = copy.deepcopy(velocity2d)
        self.pml_damping = pml_damping

    def set_velocity(self, velocity2d=Velocity2D()):

        self.vel2D = copy.deepcopy(velocity2d)

    def set_pml_damping(self, pml_damping=Common.pml_damping):

        self.pml_damping = pml_damping

    def create_matrix(self, omega=Common.omega, transpose_flag=False):

        # Check if omega is in range
        if not self.vel2D.geometry2D.omega_min <= omega <= self.vel2D.geometry2D.omega_max:
            raise ValueError("Omega outside range supported by geometry object.")

        # Create sx and sz arrays
        sx = self.__sx_array(omega=omega)
        sz = self.__sz_array(omega=omega)

        # Create lists to hold matrix entries
        data = []
        rows = []
        cols = []

        # Define some parameters
        nx = self.vel2D.geometry2D.gridpointsX - 2
        nz = self.vel2D.geometry2D.gridpointsZ - 2
        hx = self.vel2D.geometry2D.dx
        hz = self.vel2D.geometry2D.dz

        ####################################################################################################
        # Loop over interior nodes
        for i1 in range(2, nz):
            n1 = (i1 - 1) * nx
            p1z = sz[2 * i1] / hz
            p2z = sz[2 * i1 + 1] / hz
            p3z = sz[2 * i1 - 1] / hz

            for i2 in range(2, nx):
                n2 = n1 + i2 - 1
                p1x = sx[2 * i2] / hx
                p2x = sx[2 * i2 + 1] / hx
                p3x = sx[2 * i2 - 1] / hx

                rows.append(n2)
                cols.append(n2)
                data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[i2, i1]) ** 2)

                rows.append(n2)
                cols.append(n2 + 1)
                data.append(p1x * p2x)

                rows.append(n2)
                cols.append(n2 - 1)
                data.append(p1x * p3x)

                rows.append(n2)
                cols.append(n2 + nx)
                data.append(p1z * p2z)

                rows.append(n2)
                cols.append(n2 - nx)
                data.append(p1z * p3z)

        ####################################################################################################
        # Edges except corners

        # 1. Bottom
        n1 = 0
        p1z = sz[2] / hz
        p2z = sz[3] / hz
        p3z = sz[1] / hz

        for i2 in range(2, nx):
            n2 = n1 + i2 - 1
            p1x = sx[2 * i2] / hx
            p2x = sx[2 * i2 + 1] / hx
            p3x = sx[2 * i2 - 1] / hx

            rows.append(n2)
            cols.append(n2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[i2, 1]) ** 2)

            rows.append(n2)
            cols.append(n2 + 1)
            data.append(p1x * p2x)

            rows.append(n2)
            cols.append(n2 - 1)
            data.append(p1x * p3x)

            rows.append(n2)
            cols.append(n2 + nx)
            data.append(p1z * p2z)

        # 2. Top
        n1 = (nz - 1) * nx
        p1z = sz[2 * nz] / hz
        p2z = sz[2 * nz + 1] / hz
        p3z = sz[2 * nz - 1] / hz

        for i2 in range(2, nx):
            n2 = n1 + i2 - 1
            p1x = sx[2 * i2] / hx
            p2x = sx[2 * i2 + 1] / hx
            p3x = sx[2 * i2 - 1] / hx

            rows.append(n2)
            cols.append(n2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[i2, nz]) ** 2)

            rows.append(n2)
            cols.append(n2 + 1)
            data.append(p1x * p2x)

            rows.append(n2)
            cols.append(n2 - 1)
            data.append(p1x * p3x)

            rows.append(n2)
            cols.append(n2 - nx)
            data.append(p1z * p3z)

        # 3. Left
        n1 = 0
        p1x = sx[2] / hx
        p2x = sx[3] / hx
        p3x = sx[1] / hx

        for i1 in range(2, nz):
            n2 = n1 + (i1 - 1) * nx
            p1z = sz[2 * i1] / hz
            p2z = sz[2 * i1 + 1] / hz
            p3z = sz[2 * i1 - 1] / hz

            rows.append(n2)
            cols.append(n2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[1, i1]) ** 2)

            rows.append(n2)
            cols.append(n2 + 1)
            data.append(p1x * p2x)

            rows.append(n2)
            cols.append(n2 + nx)
            data.append(p1z * p2z)

            rows.append(n2)
            cols.append(n2 - nx)
            data.append(p1z * p3z)

        # 4. Right
        n1 = nx - 1
        p1x = sx[2 * nx] / hx
        p2x = sx[2 * nx + 1] / hx
        p3x = sx[2 * nx - 1] / hx

        for i1 in range(2, nz):
            n2 = n1 + (i1 - 1) * nx
            p1z = sz[2 * i1] / hz
            p2z = sz[2 * i1 + 1] / hz
            p3z = sz[2 * i1 - 1] / hz

            rows.append(n2)
            cols.append(n2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[nx, i1]) ** 2)

            rows.append(n2)
            cols.append(n2 - 1)
            data.append(p1x * p3x)

            rows.append(n2)
            cols.append(n2 + nx)
            data.append(p1z * p2z)

            rows.append(n2)
            cols.append(n2 - nx)
            data.append(p1z * p3z)

        ####################################################################################################
        # Corners

        # 1. Bottom Left
        n2 = 0
        p1z = sz[2] / hz
        p2z = sz[3] / hz
        p3z = sz[1] / hz
        p1x = sx[2] / hx
        p2x = sx[3] / hx
        p3x = sx[1] / hx

        rows.append(n2)
        cols.append(n2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[1, 1]) ** 2)

        rows.append(n2)
        cols.append(n2 + 1)
        data.append(p1x * p2x)

        rows.append(n2)
        cols.append(n2 + nx)
        data.append(p1z * p2z)

        # 2. Bottom Right
        n2 = nx - 1
        p1z = sz[2] / hz
        p2z = sz[3] / hz
        p3z = sz[1] / hz
        p1x = sx[2 * nx] / hx
        p2x = sx[2 * nx + 1] / hx
        p3x = sx[2 * nx - 1] / hx

        rows.append(n2)
        cols.append(n2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[nx, 1]) ** 2)

        rows.append(n2)
        cols.append(n2 - 1)
        data.append(p1x * p3x)

        rows.append(n2)
        cols.append(n2 + nx)
        data.append(p1z * p2z)

        # 3. Top Left
        n2 = (nz - 1) * nz
        p1z = sz[2 * nz] / hz
        p2z = sz[2 * nz + 1] / hz
        p3z = sz[2 * nz - 1] / hz
        p1x = sx[2] / hx
        p2x = sx[3] / hx
        p3x = sx[1] / hx

        rows.append(n2)
        cols.append(n2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[1, nz]) ** 2)

        rows.append(n2)
        cols.append(n2 + 1)
        data.append(p1x * p2x)

        rows.append(n2)
        cols.append(n2 - nx)
        data.append(p1z * p3z)

        # 4. Top Right
        n2 = (nz - 1) * nx + nx - 1
        p1z = sz[2 * nz] / hz
        p2z = sz[2 * nz + 1] / hz
        p3z = sz[2 * nz - 1] / hz
        p1x = sx[2 * nx] / hx
        p2x = sx[2 * nx + 1] / hx
        p3x = sx[2 * nx - 1] / hx

        rows.append(n2)
        cols.append(n2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.vel2D.vel[nx, nz]) ** 2)

        rows.append(n2)
        cols.append(n2 - 1)
        data.append(p1x * p3x)

        rows.append(n2)
        cols.append(n2 - nx)
        data.append(p1z * p3z)

        # Convert to csc format
        if transpose_flag:
            mtx = csc_matrix((data, (cols, rows)), shape=(nx * nz, nx * nz))
            mtx = mtx.conjugate()
        else:
            mtx = csc_matrix((data, (rows, cols)), shape=(nx * nz, nx * nz))

        return mtx

    def __sx_array(self, omega):

        pml_width = self.vel2D.geometry2D.ncellsX_pad * self.vel2D.geometry2D.dx
        dx = self.vel2D.geometry2D.dx / 2.0
        end_x = self.vel2D.geometry2D.dimX + 2 * pml_width
        sx = np.zeros(shape=(2 * self.vel2D.geometry2D.gridpointsX - 1,), dtype=np.complex64)

        for i1 in range(2 * self.vel2D.geometry2D.ncellsX_pad + 1):
            sx[i1] = (1.0 - i1 * dx / pml_width) ** 2

        for i1 in range(2 * (self.vel2D.geometry2D.ncellsX + self.vel2D.geometry2D.ncellsX_pad),
                        2 * self.vel2D.geometry2D.gridpointsX - 1):
            sx[i1] = (1.0 - (end_x - i1 * dx) / pml_width) ** 2

        sx = (self.pml_damping / pml_width) * sx
        sx = 1 + Common.i * sx / omega
        return 1.0 / sx

    def __sz_array(self, omega):

        pml_width = self.vel2D.geometry2D.ncellsZ_pad * self.vel2D.geometry2D.dz
        dz = self.vel2D.geometry2D.dz / 2.0
        end_z = self.vel2D.geometry2D.dimZ + 2 * pml_width
        sz = np.zeros(shape=(2 * self.vel2D.geometry2D.gridpointsZ - 1,), dtype=np.complex64)

        for i1 in range(2 * self.vel2D.geometry2D.ncellsZ_pad + 1):
            sz[i1] = (1.0 - i1 * dz / pml_width) ** 2

        for i1 in range(2 * (self.vel2D.geometry2D.ncellsZ + self.vel2D.geometry2D.ncellsZ_pad),
                        2 * self.vel2D.geometry2D.gridpointsZ - 1):
            sz[i1] = (1.0 - (end_z - i1 * dz) / pml_width) ** 2

        sz = (self.pml_damping / pml_width) * sz
        sz = 1 + Common.i * sz / omega
        return 1.0 / sz


if __name__ == "__main__":

    # Create a default Velocity 2D object
    geom2d = CreateGeometry2D(xdim=0.5, zdim=0.5, vmin=0.5, vmax=1.5, omega_max=10)
    vel2d = Velocity2D(geometry2d=geom2d)

    # Create a MatrixHelmholtz2D object
    mat_helmholtz2d = CreateMatrixHelmholtz2D(velocity2d=vel2d, pml_damping=Common.pml_damping)
    mat = mat_helmholtz2d.create_matrix(omega=5.0)
    mat1 = splu(mat)

    # Perform a solve
    ngridx = vel2d.geometry2D.gridpointsX - 2
    ngridz = vel2d.geometry2D.gridpointsZ - 2
    b = np.zeros(shape=(ngridx * ngridz, ), dtype=np.complex64)
    b[int(ngridx * ngridz / 2) + 300] = 1.0
    x = mat1.solve(b)

    # Plot x
    xreal = np.reshape(np.real(x), (ngridz, ngridx))
    ximag = np.reshape(np.imag(x), (ngridz, ngridx))
    plt.figure()
    plt.subplot(121)
    plt.title('Real(x)')
    plt.grid(color='w')
    plt.imshow(xreal)
    plt.colorbar()

    plt.subplot(122)
    plt.title('Imag(x)')
    plt.grid(color='w')
    plt.imshow(ximag)
    plt.colorbar()
    plt.show()
