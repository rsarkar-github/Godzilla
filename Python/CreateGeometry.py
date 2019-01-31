# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:04:30 2017
@author: rahul
"""
from Common import*


class CreateGeometry2D(object):
    """
    This class creates the geometry of the seismic experiment
    """
    """
    TODO:
    1. Add exception handling
    2. Add plotting capabilities to display the geometry
    """

    def __init__(
            self,
            xdim=Common.dim,
            zdim=Common.dim,
            vmin=Common.vel,
            vmax=Common.vel,
            omega_max=Common.omega,
            omega_min=3.0
    ):

        ####################################################################################################
        # These quantities cannot be changed after class is initialized
        ####################################################################################################
        self.__dimX = float(xdim)
        self.__dimZ = float(zdim)
        self.__vmin = float(vmin)
        self.__vmax = float(vmax)
        self.__omega_min = float(omega_min)
        self.__omega_max = float(omega_max)
        self.__lambda_min = 2 * Common.pi * vmin / float(omega_max)
        self.__lambda_max = 2 * Common.pi * vmax / float(omega_min)

        ####################################################################################################
        # These quantities can be changed after class is initialized
        ####################################################################################################

        # Set parameters by default
        self.__ncellsX = 0
        self.__ncellsZ = 0
        self.__dx = 0.0
        self.__dz = 0.0
        self.__ncellsX_pad = 0
        self.__ncellsZ_pad = 0
        self.__gridpointsX = 0
        self.__gridpointsZ = 0
        self.set_default_params()

    def set_default_params(self):

        self.__ncellsX = int((self.__dimX / self.__lambda_min) * Common.ppw)
        self.__ncellsZ = int((self.__dimZ / self.__lambda_min) * Common.ppw)
        self.__dx = self.__dimX / float(self.__ncellsX)
        self.__dz = self.__dimZ / float(self.__ncellsZ)
        self.__ncellsX_pad = int(Common.pml * self.__lambda_max / self.__dx)
        self.__ncellsZ_pad = int(Common.pml * self.__lambda_max / self.__dz)
        self.__gridpointsX = self.__ncellsX + 2 * self.__ncellsX_pad + 1
        self.__gridpointsZ = self.__ncellsZ + 2 * self.__ncellsZ_pad + 1

    def set_params(
            self,
            ncells_x,
            ncells_z,
            ncells_x_pad,
            ncells_z_pad,
            check=True
    ):

        if check:
            if self.__check_gridparams(self.__dimX, ncells_x, ncells_x_pad):
                if self.__check_gridparams(self.__dimZ, ncells_z, ncells_z_pad):
                    self.__ncellsX = int(ncells_x)
                    self.__ncellsZ = int(ncells_z)
                    self.__dx = self.__dimX / float(self.__ncellsX)
                    self.__dz = self.__dimZ / float(self.__ncellsZ)
                    self.__ncellsX_pad = int(ncells_x_pad)
                    self.__ncellsZ_pad = int(ncells_z_pad)
                    self.__gridpointsX = self.__ncellsX + 2 * self.__ncellsX_pad + 1
                    self.__gridpointsZ = self.__ncellsZ + 2 * self.__ncellsZ_pad + 1
                else:
                    raise ValueError("Grid parameter check failed.")
            else:
                raise ValueError("Grid parameter check failed.")
        else:
            self.__ncellsX = int(ncells_x)
            self.__ncellsZ = int(ncells_z)
            self.__dx = self.__dimX / float(self.__ncellsX)
            self.__dz = self.__dimZ / float(self.__ncellsZ)
            self.__ncellsX_pad = int(ncells_x_pad)
            self.__ncellsZ_pad = int(ncells_z_pad)
            self.__gridpointsX = self.__ncellsX + 2 * self.__ncellsX_pad + 1
            self.__gridpointsZ = self.__ncellsZ + 2 * self.__ncellsZ_pad + 1

    """
    # Properties
    """

    @property
    def dimX(self):

        return self.__dimX

    @property
    def dimZ(self):

        return self.__dimZ

    @property
    def vmin(self):

        return self.__vmin

    @property
    def vmax(self):

        return self.__vmax

    @property
    def omega_min(self):

        return self.__omega_min

    @property
    def omega_max(self):

        return self.__omega_max

    @property
    def lambda_min(self):

        return self.__lambda_min

    @property
    def lambda_max(self):

        return self.__lambda_max

    @property
    def ncellsX(self):

        return self.__ncellsX

    @property
    def ncellsZ(self):

        return self.__ncellsZ

    @property
    def dx(self):

        return self.__dx

    @property
    def dz(self):

        return self.__dz

    @property
    def ncellsX_pad(self):

        return self.__ncellsX_pad

    @property
    def ncellsZ_pad(self):

        return self.__ncellsZ_pad

    @property
    def gridpointsX(self):

        return self.__gridpointsX

    @property
    def gridpointsZ(self):

        return self.__gridpointsZ

    """
    # Private Methods
    """

    def __check_gridparams(self, dim, ncells, ncells_pad):

        d_max = self.__lambda_min / Common.ppw
        ncells_min = int((dim / self.__lambda_min) * Common.ppw)
        ncells_pad_min = int(Common.pml * self.__lambda_max / d_max)

        if ncells >= ncells_min and ncells_pad >= ncells_pad_min:
            return True
        else:
            return False
