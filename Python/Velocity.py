# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:04:30 2017
@author: rahul
"""
from CreateGeometry import*
import copy
import numpy as np
import matplotlib.pyplot as plt


class Velocity2D(object):
    """
    This class creates velocity from the geometry of the seismic experiment
    """

    def __init__(
            self,
            geometry2d=CreateGeometry2D()
    ):

        self.geometry2D = copy.deepcopy(geometry2d)
        self.vel = self.create_default_velocity()
        self.vmin = np.amin(self.vel)
        self.vmax = np.amax(self.vel)

    def create_default_velocity(self):

        vel = 0.5 * (self.geometry2D.vmin + self.geometry2D.vmax)
        return np.zeros(shape=(self.geometry2D.gridpointsX, self.geometry2D.gridpointsZ), dtype=float) + vel

    def set_geometry(self, geometry2d=CreateGeometry2D()):

        self.geometry2D = copy.deepcopy(geometry2d)
        self.vel = self.create_default_velocity()
        self.vmin = np.amin(self.vel)
        self.vmax = np.amax(self.vel)

    def set_velocity(self, velocity):

        x = np.zeros(shape=(self.geometry2D.gridpointsX, self.geometry2D.gridpointsZ), dtype=float)
        if np.all(velocity * 0 == x):
            vmin = np.amin(velocity)
            vmax = np.amax(velocity)
            if vmin >= self.geometry2D.vmin and vmax <= self.geometry2D.vmax:
                self.vel = copy.deepcopy(velocity)
                self.vmin = vmin
                self.vmax = vmax
            else:
                raise ValueError("Velocity values outside range of geometry object.")
        else:
            raise ValueError("Velocity object type inconsistent.")

    def set_constant_velocity(self, vel=Common.vel):

        if self.geometry2D.vmin <= vel <= self.geometry2D.vmax:
            self.vel = np.zeros(shape=(self.geometry2D.gridpointsX, self.geometry2D.gridpointsZ), dtype=float) + vel
            self.vmin = vel
            self.vmax = vel
        else:
            raise ValueError("Velocity values outside range of geometry object.")

    def create_gaussian_perturbation(self, dvel, sigma_x, sigma_z, nx, nz):

        pert = np.zeros(shape=(self.geometry2D.gridpointsX, self.geometry2D.gridpointsZ), dtype=float)

        for i1 in range(self.geometry2D.gridpointsX):
            for j1 in range(self.geometry2D.gridpointsZ):
                f = (float(i1 - nx) * self.geometry2D.dx / sigma_x) ** 2 \
                    + (float(j1 - nz) * self.geometry2D.dz / sigma_z) ** 2
                pert[i1, j1] = np.exp(-0.5 * f)

        pert = pert * dvel
        vel1 = self.vel + pert
        vel1_min = np.amin(vel1)
        vel1_max = np.amax(vel1)

        if vel1_min >= self.geometry2D.vmin and vel1_max <= self.geometry2D.vmax:
            self.vel = vel1
            self.vmin = vel1_min
            self.vmax = vel1_max
        else:
            raise ValueError("Perturbation leads to values outside range of geometry object.")

    def plot_nopad(
            self,
            title="Velocity",
            xlabel="X",
            ylabel="Z",
            colorlabel="km/s",
            vmin="",
            vmax="",
            savefile=""
    ):

        # Extract velocity field without padding region
        cells_x = self.geometry2D.ncellsX
        cells_z = self.geometry2D.ncellsZ
        padcells_x = self.geometry2D.ncellsX_pad
        padcells_z = self.geometry2D.ncellsZ_pad
        vel_field = self.vel[padcells_x: padcells_x + cells_x + 1, padcells_z: padcells_z + cells_z + 1]

        # Plot the velocity field
        if vmin is "":
            vmin = np.amin(vel_field)
        if vmax is "":
            vmax = np.amax(vel_field)

        plt.figure()
        plt.imshow(np.transpose(vel_field), origin="lower", vmin=vmin, vmax=vmax)
        cb = plt.colorbar()
        cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if savefile is not "":
            plt.savefig(savefile, bbox_inches="tight")

        plt.show()

    def plot(
            self,
            title="Velocity",
            xlabel="X",
            ylabel="Z",
            colorlabel="km/s",
            vmin="",
            vmax="",
            savefile=""
    ):

        # Plot the velocity field
        if vmin is "":
            vmin = np.amin(self.vel)
        if vmax is "":
            vmax = np.amax(self.vel)

        plt.figure()
        plt.imshow(np.transpose(self.vel), origin="lower", vmin=vmin, vmax=vmax)
        cb = plt.colorbar()
        cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if savefile is not "":
            plt.savefig(savefile, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    # Create a default Velocity 2D object
    geom2d = CreateGeometry2D(xdim=1.0, zdim=1.0, vmin=0.5, vmax=1.5, omega_max=50)
    vel2d = Velocity2D(geometry2d=geom2d)

    # plt.figure()
    plt.subplot(121)
    plt.imshow(vel2d.vel)
    plt.colorbar()

    # Create a perturbation
    vel2d.create_gaussian_perturbation(dvel=0.3, sigma_x=0.5, sigma_z=0.5, nx=160, nz=160)
    plt.subplot(122)
    plt.imshow(vel2d.vel)
    plt.colorbar()

    plt.show()
