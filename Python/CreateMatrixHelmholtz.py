# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:04:30 2017
@author: rahul
"""
from Common import*
from Velocity import Velocity2D
from CreateGeometry import CreateGeometry2D
from Utilities import TypeChecker
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import copy
import numpy as np


class CreateMatrixHelmholtz2D(object):
    """
    Create a sparse matrix object that can solve the Helmholtz equation.
    Currently the class only supports PML boundary conditions on all faces.
    """
    """
    TODO:
    1. Add exception handling
    2. Think about setting pml_values (exception handling)
    """

    def __init__(self, velocity2d=Velocity2D(), pml_damping=Common.pml_damping):

        TypeChecker.check(x=velocity2d, expected_type=(Velocity2D,))
        TypeChecker.check_float_positive(x=pml_damping)

        self.__vel2D = copy.deepcopy(velocity2d)
        self.__pml_damping = pml_damping

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        return self.__vel2D == other.vel2D and self.__pml_damping == other.pml_damping

    def __ne__(self, other):

        if not isinstance(other, self.__class__):
            return True

        return self.__vel2D != other.vel2D or self.__pml_damping != other.pml_damping

    def create_matrix(self, omega=None, conjugate_flag=False):

        TypeChecker.check(x=conjugate_flag, expected_type=(bool,))

        if omega is not None:

            TypeChecker.check_float_positive(x=omega)

            # Check if omega is in range
            if not self.__vel2D.geometry2D.omega_min <= omega <= self.__vel2D.geometry2D.omega_max:
                raise ValueError("Omega outside range supported by geometry object.")

        else:

            omega = Common.omega

            # Check if omega is in range
            if not self.__vel2D.geometry2D.omega_min <= omega <= self.__vel2D.geometry2D.omega_max:

                print("Default omega value outside range supported by geometry object. "
                      "Setting omega to be the mid-value supported by the geometry.")

                omega = 0.5 * (self.__vel2D.geometry2D.omega_min + self.__vel2D.geometry2D.omega_max)

        # Create sx and sz arrays
        sx = self.__sx_array(omega=omega)
        sz = self.__sz_array(omega=omega)

        # Create lists to hold matrix entries
        data = []
        rows = []
        cols = []

        # Define some parameters
        nx = self.__vel2D.geometry2D.gridpointsX - 2
        nz = self.__vel2D.geometry2D.gridpointsZ - 2
        hx = self.__vel2D.geometry2D.dx
        hz = self.__vel2D.geometry2D.dz

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
                data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[i2, i1]) ** 2)

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
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[i2, 1]) ** 2)

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
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[i2, nz]) ** 2)

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
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[1, i1]) ** 2)

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
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[nx, i1]) ** 2)

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
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[1, 1]) ** 2)

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
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[nx, 1]) ** 2)

        rows.append(n2)
        cols.append(n2 - 1)
        data.append(p1x * p3x)

        rows.append(n2)
        cols.append(n2 + nx)
        data.append(p1z * p2z)

        # 3. Top Left
        n2 = (nz - 1) * nx
        p1z = sz[2 * nz] / hz
        p2z = sz[2 * nz + 1] / hz
        p3z = sz[2 * nz - 1] / hz
        p1x = sx[2] / hx
        p2x = sx[3] / hx
        p3x = sx[1] / hx

        rows.append(n2)
        cols.append(n2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[1, nz]) ** 2)

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
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / self.__vel2D.vel[nx, nz]) ** 2)

        rows.append(n2)
        cols.append(n2 - 1)
        data.append(p1x * p3x)

        rows.append(n2)
        cols.append(n2 - nx)
        data.append(p1z * p3z)

        # Convert to csc format
        if conjugate_flag:
            mtx = csc_matrix((data, (cols, rows)), shape=(nx * nz, nx * nz))
            mtx = mtx.conjugate()
        else:
            mtx = csc_matrix((data, (rows, cols)), shape=(nx * nz, nx * nz))

        return mtx

    """
    # Properties
    """

    @property
    def vel2D(self):

        return self.__vel2D

    @vel2D.setter
    def vel2D(self, velocity2d):

        TypeChecker.check(x=velocity2d, expected_type=(Velocity2D,))
        self.__vel2D = copy.deepcopy(velocity2d)

    @property
    def pml_damping(self):

        return self.__pml_damping

    @pml_damping.setter
    def pml_damping(self, pml_damping_value):

        TypeChecker.check_float_positive(x=pml_damping_value)
        self.__pml_damping = pml_damping_value

    """
    # Private Methods
    """

    def __sx_array(self, omega):

        pml_width = self.__vel2D.geometry2D.ncellsX_pad * self.__vel2D.geometry2D.dx
        dx = self.__vel2D.geometry2D.dx / 2.0
        end_x = self.__vel2D.geometry2D.dimX + 2 * pml_width
        sx = np.zeros(shape=(2 * self.__vel2D.geometry2D.gridpointsX - 1,), dtype=np.complex64)

        for i1 in range(2 * self.__vel2D.geometry2D.ncellsX_pad + 1):
            sx[i1] = (1.0 - i1 * dx / pml_width) ** 2

        for i1 in range(2 * (self.__vel2D.geometry2D.ncellsX + self.__vel2D.geometry2D.ncellsX_pad),
                        2 * self.__vel2D.geometry2D.gridpointsX - 1):
            sx[i1] = (1.0 - (end_x - i1 * dx) / pml_width) ** 2

        sx = (self.__pml_damping / pml_width) * sx
        sx = 1 + Common.i * sx / omega
        return 1.0 / sx

    def __sz_array(self, omega):

        pml_width = self.__vel2D.geometry2D.ncellsZ_pad * self.__vel2D.geometry2D.dz
        dz = self.__vel2D.geometry2D.dz / 2.0
        end_z = self.__vel2D.geometry2D.dimZ + 2 * pml_width
        sz = np.zeros(shape=(2 * self.__vel2D.geometry2D.gridpointsZ - 1,), dtype=np.complex64)

        for i1 in range(2 * self.__vel2D.geometry2D.ncellsZ_pad + 1):
            sz[i1] = (1.0 - i1 * dz / pml_width) ** 2

        for i1 in range(2 * (self.__vel2D.geometry2D.ncellsZ + self.__vel2D.geometry2D.ncellsZ_pad),
                        2 * self.__vel2D.geometry2D.gridpointsZ - 1):
            sz[i1] = (1.0 - (end_z - i1 * dz) / pml_width) ** 2

        sz = (self.__pml_damping / pml_width) * sz
        sz = 1 + Common.i * sz / omega
        return 1.0 / sz


if __name__ == "__main__":

    # Create a default Velocity 2D object
    geom2d = CreateGeometry2D(xdim=6.0, zdim=2.5, vmin=0.5, vmax=1.5, omega_max=20)
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
    plt.imshow(xreal, cmap="jet")
    plt.colorbar()

    plt.subplot(122)
    plt.title('Imag(x)')
    plt.grid(color='w')
    plt.imshow(ximag, cmap="jet")
    plt.colorbar()
    plt.show()
