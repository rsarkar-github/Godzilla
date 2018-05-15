# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:04:30 2017
@author: rahul
"""
from Common import Common


class CreateGeometry2D(object):
    """
    This class creates the geometry of the seismic experiment
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
        self.dimX = float(xdim)
        self.dimZ = float(zdim)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.omega_max = float(omega_max)
        self.omega_min = float(omega_min)
        self.lambda_min = 2 * Common.pi * vmin / float(omega_max)
        self.lambda_max = 2 * Common.pi * vmax / float(omega_min)

        ####################################################################################################
        # These quantities can be changed after class is initialized
        ####################################################################################################

        # Set parameters by default
        self.ncellsX = 0
        self.ncellsZ = 0
        self.dx = 0.0
        self.dz = 0.0
        self.ncellsX_pad = 0
        self.ncellsZ_pad = 0
        self.gridpointsX = 0
        self.gridpointsZ = 0
        self.set_default_params()

        # Set source positions
        self.sources = []
        self.set_default_sources()

        # Set receiver positions
        self.receivers = []
        self.set_default_receivers()

    def set_default_params(self):

        self.ncellsX = int((self.dimX / self.lambda_min) * Common.ppw)
        self.ncellsZ = int((self.dimZ / self.lambda_min) * Common.ppw)
        self.dx = self.dimX / float(self.ncellsX)
        self.dz = self.dimZ / float(self.ncellsZ)
        self.ncellsX_pad = int(Common.pml * self.lambda_max / self.dx)
        self.ncellsZ_pad = int(Common.pml * self.lambda_max / self.dz)
        self.gridpointsX = self.ncellsX + 2 * self.ncellsX_pad + 1
        self.gridpointsZ = self.ncellsZ + 2 * self.ncellsZ_pad + 1

    def set_params(
            self,
            ncells_x,
            ncells_z,
            ncells_x_pad,
            ncells_z_pad,
            check=True
    ):

        if check:
            if self.check_gridparams(self.dimX, ncells_x, ncells_x_pad):
                if self.check_gridparams(self.dimZ, ncells_z, ncells_z_pad):
                    self.ncellsX = int(ncells_x)
                    self.ncellsZ = int(ncells_z)
                    self.dx = self.dimX / float(self.ncellsX)
                    self.dz = self.dimZ / float(self.ncellsZ)
                    self.ncellsX_pad = int(ncells_x_pad)
                    self.ncellsZ_pad = int(ncells_z_pad)
                    self.gridpointsX = self.ncellsX + 2 * self.ncellsX_pad + 1
                    self.gridpointsZ = self.ncellsZ + 2 * self.ncellsZ_pad + 1
                else:
                    raise ValueError("Grid parameter check failed.")
            else:
                raise ValueError("Grid parameter check failed.")
        else:
            self.ncellsX = int(ncells_x)
            self.ncellsZ = int(ncells_z)
            self.dx = self.dimX / float(self.ncellsX)
            self.dz = self.dimZ / float(self.ncellsZ)
            self.ncellsX_pad = int(ncells_x_pad)
            self.ncellsZ_pad = int(ncells_z_pad)
            self.gridpointsX = self.ncellsX + 2 * self.ncellsX_pad + 1
            self.gridpointsZ = self.ncellsZ + 2 * self.ncellsZ_pad + 1

    def check_gridparams(self, dim, ncells, ncells_pad):

        d_max = self.lambda_min / Common.ppw
        ncells_min = int((dim / self.lambda_min) * Common.ppw)
        ncells_pad_min = int(Common.pml * self.lambda_max / d_max)

        if ncells >= ncells_min and ncells_pad >= ncells_pad_min:
            return True
        else:
            return False

    def set_default_sources(self, skip=1):

        self.sources = [[self.ncellsX_pad + i, self.ncellsZ_pad] for i in range(0, self.ncellsX + 1, skip)]

    def set_default_receivers(self, skip=1):

        self.receivers = [[self.ncellsX_pad + i, self.ncellsZ_pad] for i in range(0, self.ncellsX + 1, skip)]
