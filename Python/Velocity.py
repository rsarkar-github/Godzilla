# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:04:30 2017
@author: rahul
"""
from Common import*
from CreateGeometry import CreateGeometry2D
from Utilities import TypeChecker
import copy
import numpy as np


class Velocity2D(object):
    """
    This class creates velocity from the geometry of the seismic experiment
    """
    """
    TODO:
    1. Add exception handling (e.g. what happens in constructor when velocity is provided but gepmetry2d is not?)
    """

    def __init__(
            self,
            geometry2d=CreateGeometry2D(),
            velocity=None
    ):

        TypeChecker.check(x=geometry2d, expected_type=(CreateGeometry2D,))
        self.__geometry2D = geometry2d

        if velocity is None:

            self.__vel = self.create_default_velocity()
            self.__vmin = np.amin(self.__vel)
            self.__vmax = np.amax(self.__vel)

        else:

            TypeChecker.check(x=velocity, expected_type=(np.ndarray,))

            x = np.zeros(shape=(self.__geometry2D.gridpointsX, self.__geometry2D.gridpointsZ), dtype=float)
            if np.all(velocity * 0 == x):
                vmin = np.amin(velocity)
                vmax = np.amax(velocity)
                if vmin >= self.__geometry2D.vmin and vmax <= self.__geometry2D.vmax:
                    self.__vel = copy.deepcopy(velocity)
                    self.__vmin = vmin
                    self.__vmax = vmax
                else:
                    raise ValueError("Velocity values outside range of geometry object.")
            else:
                raise ValueError("Velocity object type inconsistent.")

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        return self.__geometry2D == other.geometry2D \
            and np.all(self.__vel == other.vel) \
            and self.__vmin == other.vmin \
            and self.__vmax == other.vmax

    def __ne__(self, other):

        if not isinstance(other, self.__class__):
            return True

        return self.__geometry2D != other.geometry2D \
            or np.any(self.__vel != other.vel) \
            or self.__vmin != other.vmin \
            or self.__vmax != other.vmax

    def create_default_velocity(self):

        vel = 0.5 * (self.geometry2D.vmin + self.geometry2D.vmax)
        return np.zeros(shape=(self.geometry2D.gridpointsX, self.geometry2D.gridpointsZ), dtype=float) + vel

    def set_constant_velocity(self, vel=Common.vel):

        if self.__geometry2D.vmin <= vel <= self.__geometry2D.vmax:
            self.__vel = vel + np.zeros(
                shape=(self.__geometry2D.gridpointsX, self.__geometry2D.gridpointsZ),
                dtype=float
            )
            self.__vmin = vel
            self.__vmax = vel
        else:
            raise ValueError("Velocity values outside range of geometry object.")

    def create_gaussian_perturbation(self, dvel, sigma_x, sigma_z, nx, nz):

        pert = np.zeros(shape=(self.__geometry2D.gridpointsX, self.__geometry2D.gridpointsZ), dtype=float)

        dx_by_sigma_x = self.__geometry2D.dx / sigma_x
        dz_by_sigma_z = self.__geometry2D.dz / sigma_z

        for i1 in range(self.__geometry2D.gridpointsX):
            for j1 in range(self.__geometry2D.gridpointsZ):
                f = (float(i1 - nx) * dx_by_sigma_x) ** 2 + (float(j1 - nz) * dz_by_sigma_z) ** 2
                pert[i1, j1] = np.exp(-0.5 * f)

        self.vel = self.__vel + pert * dvel

    def plot_difference(
            self,
            vel_other,
            pad=True,
            title="Velocity Difference",
            xlabel="X",
            ylabel="Z",
            colorlabel="km/s",
            vmin="",
            vmax="",
            cmap="jet",
            show=True,
            savefile=""
    ):

        TypeChecker.check(x=vel_other, expected_type=(Velocity2D,))

        if self.__geometry2D == vel_other.geometry2D:
            vel_diff = self.__vel - vel_other.vel

        else:
            raise TypeError("Velocities being compared to do have same geometries.")

        if pad:

            # Plot the velocity field difference
            if vmin is "":
                vmin = np.amin(vel_diff)
            if vmax is "":
                vmax = np.amax(vel_diff)

            plt.figure()
            plt.imshow(np.transpose(vel_diff), origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
            cb = plt.colorbar()
            cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            if savefile is not "":
                plt.savefig(savefile, bbox_inches="tight")

            if show:
                plt.show()

        else:

            # Extract velocity field without padding region
            cells_x = self.__geometry2D.ncellsX
            cells_z = self.__geometry2D.ncellsZ
            padcells_x = self.__geometry2D.ncellsX_pad
            padcells_z = self.__geometry2D.ncellsZ_pad
            vel_diff = vel_diff[padcells_x: padcells_x + cells_x + 1, padcells_z: padcells_z + cells_z + 1]

            # Plot the velocity field difference
            if vmin is "":
                vmin = np.amin(vel_diff)
            if vmax is "":
                vmax = np.amax(vel_diff)

            plt.figure()
            plt.imshow(np.transpose(vel_diff), origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
            cb = plt.colorbar()
            cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            if savefile is not "":
                plt.savefig(savefile, bbox_inches="tight")

            if show:
                plt.show()

    def plot(
            self,
            title="Velocity",
            pad=True,
            xlabel="X",
            ylabel="Z",
            colorlabel="km/s",
            vmin="",
            vmax="",
            cmap="jet",
            show=True,
            savefile=""
    ):

        if pad:

            # Plot the velocity field
            if vmin is "":
                vmin = np.amin(self.__vel)
            if vmax is "":
                vmax = np.amax(self.__vel)

            plt.figure()
            plt.imshow(np.transpose(self.__vel), origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
            cb = plt.colorbar()
            cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            if savefile is not "":
                plt.savefig(savefile, bbox_inches="tight")

            if show:
                plt.show()

        else:

            # Extract velocity field without padding region
            vel_nopad = self.vel_nopad

            # Plot the velocity field
            if vmin is "":
                vmin = np.amin(vel_nopad)
            if vmax is "":
                vmax = np.amax(vel_nopad)

            plt.figure()
            plt.imshow(np.transpose(vel_nopad), origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
            cb = plt.colorbar()
            cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            if savefile is not "":
                plt.savefig(savefile, bbox_inches="tight")

            if show:
                plt.show()

    """
    # Properties
    """
    @property
    def geometry2D(self):

        return self.__geometry2D

    @geometry2D.setter
    def geometry2D(self, geometry2d):

        TypeChecker.check(x=geometry2d, expected_type=(CreateGeometry2D,))

        self.__geometry2D = geometry2d
        self.__vel = self.create_default_velocity()
        self.__vmin = np.amin(self.__vel)
        self.__vmax = np.amax(self.__vel)

    @property
    def vel(self):

        return self.__vel

    @vel.setter
    def vel(self, velocity):

        TypeChecker.check(x=velocity, expected_type=(np.ndarray,))

        x = np.zeros(shape=(self.__geometry2D.gridpointsX, self.__geometry2D.gridpointsZ), dtype=float)
        if np.all(velocity * 0 == x):
            vmin = np.amin(velocity)
            vmax = np.amax(velocity)
            if vmin >= self.__geometry2D.vmin and vmax <= self.__geometry2D.vmax:
                self.__vel = copy.deepcopy(velocity)
                self.__vmin = vmin
                self.__vmax = vmax
            else:
                raise ValueError("Velocity values outside range of geometry object.")
        else:
            raise ValueError("Velocity object type inconsistent.")

    @property
    def vel_nopad(self):

        # Extract velocity field without padding region
        cells_x = self.__geometry2D.ncellsX
        cells_z = self.__geometry2D.ncellsZ
        padcells_x = self.__geometry2D.ncellsX_pad
        padcells_z = self.__geometry2D.ncellsZ_pad
        vel_nopad = self.__vel[padcells_x: padcells_x + cells_x + 1, padcells_z: padcells_z + cells_z + 1]

        return vel_nopad

    @vel_nopad.setter
    def vel_nopad(self, velocity):

        TypeChecker.check(x=velocity, expected_type=(np.ndarray,))

        cells_x = self.__geometry2D.ncellsX
        cells_z = self.__geometry2D.ncellsZ
        padcells_x = self.__geometry2D.ncellsX_pad
        padcells_z = self.__geometry2D.ncellsZ_pad

        x = np.zeros(shape=(cells_x + 1, cells_z + 1), dtype=float)
        if np.all(velocity * 0 == x):
            vmin = np.amin(velocity)
            vmax = np.amax(velocity)
            if vmin >= self.__geometry2D.vmin and vmax <= self.__geometry2D.vmax:
                self.__vel[padcells_x: padcells_x + cells_x + 1, padcells_z: padcells_z + cells_z + 1] = velocity
                self.__vmin = np.amin(self.__vel)
                self.__vmax = np.amax(self.__vel)

            else:
                raise ValueError("Velocity values outside range of geometry object.")
        else:
            raise ValueError("Velocity object type inconsistent.")

    @property
    def vmin(self):

        return self.__vmin

    @property
    def vmax(self):

        return self.__vmax


if __name__ == "__main__":
    # Create a default Velocity 2D object
    geom2d = CreateGeometry2D(xdim=1.0, zdim=1.0, vmin=0.5, vmax=1.5, omega_max=50)
    vel2d = Velocity2D(geometry2d=geom2d)

    # plt.figure()
    plt.subplot(121)
    plt.imshow(vel2d.vel, vmin=0.5, vmax=1.5, cmap="jet")
    plt.colorbar()

    # Create a perturbation
    vel2d.create_gaussian_perturbation(dvel=0.3, sigma_x=0.5, sigma_z=0.5, nx=500, nz=500)
    plt.subplot(122)
    plt.imshow(vel2d.vel, vmin=0.5, vmax=1.5, cmap="jet")
    plt.colorbar()

    plt.show()
