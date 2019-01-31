# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:04:30 2019
@author: rahul
"""
from Common import*
from CreateGeometry import CreateGeometry2D
from Acquisition import Acquisition2D
from Velocity import Velocity2D
from CreateMatrixHelmholtz import CreateMatrixHelmholtz2D
from Utilities import TypeChecker, WaveletTools, LinearSolvers
import copy
import numpy as np
from scipy.sparse.linalg import splu
import time


class TfwiLeastSquares2D(object):
    """
    Create a Tfwi problem object for 2D test problems
    """
    """
    TODO:
    1. Add exception handling
    """
    def __init__(
            self,
            veltrue,
            velstart,
            acquisition
    ):
        # Check types
        TypeChecker.check(x=veltrue, expected_type=Velocity2D)
        TypeChecker.check(x=velstart, expected_type=Velocity2D)
        TypeChecker.check(x=acquisition, expected_type=Acquisition2D)

        # Check if geometries match
        if veltrue.geometry2D != velstart.geometry2D or veltrue.geometry2D != acquisition.geometry2D:
            raise TypeError("Geometries of veltrue, velstart and acquisition do not match.")

        # Copy the objects
        self.__geometry2D = copy.deepcopy(veltrue.geometry2D)
        self.__veltrue = copy.deepcopy(veltrue)
        self.__velstart = copy.deepcopy(velstart)
        self.__acquisition = copy.deepcopy(acquisition)

        # Set frequencies for the problem
        self.__omega_list = [
            self.__geometry2D.omega_min,
            self.__geometry2D.omega_max
        ]

        # Set wavelet coefficients
        self.__wavelet = []
        self.set_ricker_wavelet(omega_peak=self.__geometry2D.omega_max / 2.0)

    def set_constant_starting_model(self, value=None):

        if value is None:

            value = (self.__geometry2D.vmax + self.__geometry2D.vmin) / 2.0
            self.__velstart.set_constant_velocity(value)
            self.__run_cleanup(flag=3)

        else:

            TypeChecker.check_float_bounds(x=value, lb=self.__geometry2D.vmin, ub=self.__geometry2D.vmax)
            self.__velstart.set_constant_velocity(value)
            self.__run_cleanup(flag=3)

    def set_ricker_wavelet(self, omega_peak=None):

        if omega_peak is None:
            omega_peak = self.__geometry2D.omega_max / 2.0
        else:
            TypeChecker.check_float_positive(x=omega_peak)

        self.__wavelet = [
            WaveletTools.ricker_wavelet_coefficient(
                omega=omega,
                omega_peak=omega_peak
            ) for omega in self.__omega_list
        ]
        self.__run_cleanup(flag=6)

    def set_flat_spectrum_wavelet(self):

        self.__wavelet = [np.complex64(1.0) for _ in self.__omega_list]
        self.__run_cleanup(flag=6)

    ###########################################################################################################

    def perform_lsm_cg(
            self,
            save_lsm_adjoint_image=True,
            save_lsm_adjoint_allimages=False,
            lsm_adjoint_image_file="lsm_adjoint_image"
    ):

        print("Starting the conjugate gradient least squares migration...\n")
        ####################################################################################################
        # Get quantities needed throughout the computation
        ####################################################################################################

        # Get shots, receivers
        rcvlist = self.veltrue.geometry2D.receivers
        srclist = self.veltrue.geometry2D.sources
        nrcv = len(rcvlist)
        nsrc = len(srclist)

        # Get grid point info
        nx_solver = self.veltrue.geometry2D.gridpointsX - 2
        nz_solver = self.veltrue.geometry2D.gridpointsZ - 2
        nx_nopad = self.veltrue.geometry2D.ncellsX + 1
        nz_nopad = self.veltrue.geometry2D.ncellsZ + 1

        ####################################################################################################
        # Generate true data
        ####################################################################################################
        print("Generating true data...\n")

        # Create Helmholtz object
        helmholtz_true = CreateMatrixHelmholtz2D(velocity2d=self.veltrue, pml_damping=Common.pml_damping)

        # Allocate space for storing true data
        data_true = np.zeros(shape=(len(self.omega_list), nsrc, nrcv), dtype=np.complex64)

        # Loop over frequencies
        print("Looping over frequencies...\n")
        cum_time_datagen = 0.0
        for nomega, omega in enumerate(self.omega_list):

            print("Starting computation for nomega = ", nomega, " / ", len(self.omega_list) - 1)
            time_datagen_start = time.time()

            # Factorize matrices
            print("Creating factorized matrix...")
            time_fac_start = time.time()
            mat_helmholtz2d_true = helmholtz_true.create_matrix(omega)
            fac_mat_helmholtz2d_true = splu(mat_helmholtz2d_true)
            time_fac_end = time.time()
            print("Matrix factorized. Time for factorization = ", time_fac_end - time_fac_start, " s")

            # Initialize rhs
            b = np.zeros(shape=(nx_solver * nz_solver, 1), dtype=np.complex64)

            # Start cumulative time counter
            cum_time_solve = 0.0

            # Loop over shots
            for nshot, shot in enumerate(srclist):

                time_solve_start = time.time()

                # Create RHS
                b = b * 0
                shot_grid_index = nx_solver * (shot[1] - 1) + shot[0] - 1
                b[shot_grid_index] = self.wavelet[nomega]

                # Solve for wave field
                u = fac_mat_helmholtz2d_true.solve(rhs=b)

                # Sample wave field at receivers
                for nreceiver, receiver in enumerate(rcvlist):
                    receiver_grid_index = nx_solver * (receiver[1] - 1) + receiver[0] - 1
                    data_true[nomega, nshot, nreceiver] = u[receiver_grid_index]

                time_solve_end = time.time()
                cum_time_solve = cum_time_solve + (time_solve_end - time_solve_start)
                print("Solving equation for nshot = ", nshot, " / ", nsrc - 1,
                      ", Time to solve = ", time_solve_end - time_solve_start, " s")

            time_datagen_end = time.time()
            cum_time_datagen = cum_time_datagen + (time_datagen_end - time_datagen_start)

            # Print average time to solution
            print("Solving linear system for ", nsrc, "shots took ", cum_time_solve, " s")
            print("Average time for linear solve = ", cum_time_solve / float(nsrc), " s")

        # Print total data generation time
        print("Data generation for ", len(self.omega_list), " frequencies, and ",
              nsrc, " shots took ", cum_time_datagen, "s\n\n")

        # Free resources
        del helmholtz_true

        ####################################################################################################
        # Pre-compute quantities
        ####################################################################################################
        print("Starting pre-compute phase...\n")

        # //////////////////////////////////////////////////////////////////////////////////////////////////
        print("Pre-computing factorized matrices...")
        helmholtz_start = CreateMatrixHelmholtz2D(velocity2d=self.velstart, pml_damping=Common.pml_damping)

        # Initialize list for pre-computed LU factors
        fac_mat_helmholtz2d_start_list = []

        # Loop over frequencies
        print("Looping over frequencies...\n")
        cum_time_matfac = 0.0
        for nomega, omega in enumerate(self.omega_list):

            print("Starting computation for nomega = ", nomega, " / ", len(self.omega_list) - 1)

            # Factorize matrices
            print("Creating factorized matrix...")
            time_fac_start = time.time()

            mat_helmholtz2d_start = helmholtz_start.create_matrix(omega)
            fac_mat_helmholtz2d_start = splu(mat_helmholtz2d_start)
            mat_helmholtz2d_start = helmholtz_start.create_matrix(omega, transpose_flag=True)
            fac_mat_helmholtz2d_t_start = splu(mat_helmholtz2d_start)

            time_fac_end = time.time()
            fac_mat_helmholtz2d_start_list.append([fac_mat_helmholtz2d_start, fac_mat_helmholtz2d_t_start])

            print("Matrix factorized. Time for factorization of matrix and its transpose = ",
                  time_fac_end - time_fac_start, " s")
            cum_time_matfac = cum_time_matfac + (time_fac_end - time_fac_start)

        # Print total time for matrix factorization
        print("Matrix factorization for ", len(self.omega_list), " frequencies took ", cum_time_matfac, "s\n\n")

        ####################################################################################################
        # Compute rhs
        ####################################################################################################
        print("Computing rhs...\n")

        # Initialize rhs vector
        rhs = np.zeros(shape=(len(self.omega_list) * nx_nopad * nz_nopad), dtype=np.complex64)

        # Loop over frequencies
        print("Looping over frequencies...\n")
        cum_time_rhsgen = 0.0
        for nomega in range(len(self.omega_list)):

            print("Starting rhs computation for nomega = ", nomega, " / ", len(self.omega_list) - 1)
            time_rhsgen_start = time.time()

            # Initialize rhs, sampled data vector at receivers
            b = np.zeros(shape=(nx_solver * nz_solver), dtype=np.complex64)
            data_rcv = np.zeros(shape=nrcv, dtype=np.complex64)

            # Start cumulative time counter
            cum_time_solve = 0.0

            # Loop over shots
            for nshot, shot in enumerate(srclist):

                time_solve_start = time.time()

                # Create RHS
                b = b * 0
                shot_grid_index = nx_solver * (shot[1] - 1) + shot[0] - 1
                b[shot_grid_index] = self.wavelet[nomega]

                # Solve for wave field
                u = fac_mat_helmholtz2d_start_list[nomega][0].solve(rhs=b)

                # Sample wave field at receivers
                for nreceiver, receiver in enumerate(rcvlist):
                    receiver_grid_index = nx_solver * (receiver[1] - 1) + receiver[0] - 1
                    data_rcv[nreceiver] = u[receiver_grid_index]

                # Calculate residual and backproject
                data_rcv = data_true[nomega, nshot, :] - data_rcv
                b = self.__residual_2_modeling_grid(vec_residual=data_rcv, vec_out=b)
                b = fac_mat_helmholtz2d_start_list[nomega][1].solve(rhs=b)
                b = b * np.conjugate(u)

                # Add to rhs
                rhs[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad] = \
                    self.__modeling_grid_2_nopad_grid(
                        vec_model_grid=b,
                        vec_nopad_grid=rhs[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad],
                        add_flag=True
                    )

                time_solve_end = time.time()
                cum_time_solve = cum_time_solve + (time_solve_end - time_solve_start)
                print("Creating rhs for nshot = ", nshot, " / ", nsrc - 1,
                      ", Time to solve = ", time_solve_end - time_solve_start, " s")

            time_rhsgen_end = time.time()
            cum_time_rhsgen = cum_time_rhsgen + (time_rhsgen_end - time_rhsgen_start)

            # Print average time to solution
            print("Creating rhs for ", nsrc, "shots took ", cum_time_solve, " s")
            print("Average time per shot = ", cum_time_solve / float(nsrc), " s")

        # Print total rhs generation time
        print("Rhs generation for ", len(self.omega_list), " frequencies, and ",
              nsrc, " shots took ", cum_time_rhsgen, "s\n\n")

        # Plot rhs for QC
        if save_lsm_adjoint_allimages:
            for nomega in range(len(self.omega_list)):
                self.__plot_nopad_vec_complex(
                    vec=rhs[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad],
                    title1="Real",
                    title2="Imag",
                    colorbar=False,
                    show=False,
                    cmap1="jet",
                    cmap2="jet",
                    savefile=lsm_adjoint_image_file + "-" + str(nomega) + ".pdf"
                )

        # Form adjoint image
        if save_lsm_adjoint_image:
            lsm_adjoint = np.zeros(shape=(nx_nopad * nz_nopad), dtype=np.complex64)
            for nomega in range(len(self.omega_list)):
                lsm_adjoint += rhs[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad]
            self.__plot_nopad_vec_real(
                vec=np.real(lsm_adjoint),
                title="LSM adjoint image",
                colorbar=False,
                show=False,
                cmap="jet",
                savefile=lsm_adjoint_image_file + ".pdf"
            )

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
        self.__run_cleanup(flag=1)

    @property
    def veltrue(self):

        return self.__veltrue

    @veltrue.setter
    def veltrue(self, velocity2d):

        TypeChecker.check(x=velocity2d, expected_type=(Velocity2D,))

        if self.__geometry2D == velocity2d.geometry2D:

            self.__veltrue = velocity2d
            self.__run_cleanup(flag=2)

        else:
            raise TypeError("Assignment not possible as geometries do not match.")

    @property
    def veltrue_vals(self):

        return self.__veltrue.vel

    @veltrue_vals.setter
    def veltrue_vals(self, velocity):

        self.__veltrue.vel = velocity
        self.__run_cleanup(flag=2)

    @property
    def velstart(self):

        return self.__velstart

    @velstart.setter
    def velstart(self, velocity2d):

        TypeChecker.check(x=velocity2d, expected_type=(Velocity2D,))

        if self.__geometry2D == velocity2d.geometry2D:

            self.__velstart = velocity2d
            self.__run_cleanup(flag=3)

        else:
            raise TypeError("Assignment not possible as geometries do not match.")

    @property
    def velstart_vals(self):

        return self.__velstart.vel

    @velstart_vals.setter
    def velstart_vals(self, velocity):

        self.__velstart.vel = velocity
        self.__run_cleanup(flag=3)

    @property
    def acquisition(self):

        return self.__acquisition

    @acquisition.setter
    def acquisition(self, acquisition2d):

        TypeChecker.check(x=acquisition2d, expected_type=(Acquisition2D,))

        if self.__geometry2D == acquisition2d.geometry2D:

            self.__acquisition = acquisition2d
            self.__run_cleanup(flag=4)

        else:
            raise TypeError("Assignment not possible as geometries do not match.")

    @property
    def omega_list(self):

        return self.__omega_list

    @omega_list.setter
    def omega_list(self, omega):

        TypeChecker.check(x=omega, expected_type=(list,))

        omega_minimum = self.__geometry2D.omega_min
        omega_maximum = self.__geometry2D.omega_max

        for item in omega:
            TypeChecker.check_float_bounds(x=item, lb=omega_minimum, ub=omega_maximum)

        self.__omega_list = omega
        self.__run_cleanup(flag=5)

    @property
    def wavelet(self):

        return self.__wavelet

    @wavelet.setter
    def wavelet(self, wavelet_val):

        if wavelet_val == "ricker" or wavelet_val == "Ricker":
            self.set_ricker_wavelet()
            self.__run_cleanup(flag=6)
            return

        if wavelet_val == "flat":
            self.set_flat_spectrum_wavelet()
            self.__run_cleanup(flag=6)
            return

        TypeChecker.check(x=wavelet_val, expected_type=(list,))

        if len(wavelet_val) != len(self.__omega_list):
            raise TypeError("Wavelet value list does not have same length as list of frequencies.")

        for item in wavelet_val:
            TypeChecker.check(x=item, expected_type=(np.complex64, float, int))

        self.__wavelet = np.complex64(wavelet_val)
        self.__run_cleanup(flag=6)

    """
    # Private methods
    """

    def __run_cleanup(self, flag):

        """
        Performs cleanup on the class
        """

        if flag == 1:
            """
            If self.__geometry2D is reset
            """
            self.__veltrue = Velocity2D(geometry2d=self.__geometry2D)
            self.__velstart = Velocity2D(geometry2d=self.__geometry2D)
            self.__acquisition = Acquisition2D(geometry2d=self.__geometry2D)
            self.__omega_list = [self.__geometry2D.omega_min, self.__geometry2D.omega_max]
            self.set_ricker_wavelet(omega_peak=self.__geometry2D.omega_max / 2.0)

        if flag == 2:
            """
            If self.__veltrue is reset
            """
            raise NotImplementedError

        if flag == 3:
            """
            If self.__velstart is reset
            """
            raise NotImplementedError

        if flag == 4:
            """
            If self.__acquisition is reset
            """
            raise NotImplementedError

        if flag == 5:
            """
            If self.__omega_list is reset
            """
            self.set_ricker_wavelet(omega_peak=self.__geometry2D.omega_max / 2.0)

        if flag == 6:
            """
            If self.__wavelet is reset
            """
            raise NotImplementedError

    def __plot_nopad_vec_real(
            self,
            vec,
            title="Field",
            xlabel="X",
            ylabel="Z",
            colorbar=True,
            colorlabel="",
            vmin="",
            vmax="",
            cmap="hot",
            show=True,
            savefile=""
    ):

        # Plot the velocity field
        if vmin is "":
            vmin = np.amin(vec)
        if vmax is "":
            vmax = np.amax(vec)

        # Reshape vec
        vec1 = np.reshape(a=vec, newshape=(self.veltrue.geometry2D.ncellsZ + 1, self.veltrue.geometry2D.ncellsX + 1))

        plt.figure()
        plt.imshow(vec1, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if savefile is not "":
            plt.savefig(savefile, bbox_inches="tight")

        if show:
            plt.show()

    def __plot_pad_vec_real(
            self,
            vec,
            title="Field",
            xlabel="X",
            ylabel="Z",
            colorbar=True,
            colorlabel="",
            vmin="",
            vmax="",
            cmap="hot",
            show=True,
            savefile=""
    ):

        # Plot the velocity field
        if vmin is "":
            vmin = np.amin(vec)
        if vmax is "":
            vmax = np.amax(vec)

        # Reshape vec
        vec1 = np.reshape(
            a=vec,
            newshape=(self.veltrue.geometry2D.gridpointsZ - 2, self.veltrue.geometry2D.gridpointsX - 2)
        )

        plt.figure()
        plt.imshow(vec1, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if savefile is not "":
            plt.savefig(savefile, bbox_inches="tight")

        if show:
            plt.show()

    def __plot_nopad_vec_complex(
            self,
            vec,
            title1="Field",
            title2="Field",
            xlabel1="X",
            ylabel1="Z",
            xlabel2="X",
            ylabel2="Z",
            colorbar=True,
            colorlabel1="",
            colorlabel2="",
            vmin1="",
            vmax1="",
            vmin2="",
            vmax2="",
            cmap1="hot",
            cmap2="hot",
            show=True,
            savefile=""
    ):

        # Plot the velocity field
        if vmin1 is "":
            vmin1 = np.amin(np.real(vec))
        if vmax1 is "":
            vmax1 = np.amax(np.real(vec))

        if vmin2 is "":
            vmin2 = np.amin(np.imag(vec))
        if vmax2 is "":
            vmax2 = np.amax(np.imag(vec))

        # Reshape vec
        vec1 = np.reshape(
            a=np.real(vec),
            newshape=(self.veltrue.geometry2D.ncellsZ + 1, self.veltrue.geometry2D.ncellsX + 1)
        )
        vec2 = np.reshape(
            a=np.imag(vec),
            newshape=(self.veltrue.geometry2D.ncellsZ + 1, self.veltrue.geometry2D.ncellsX + 1)
        )

        plt.figure()
        plt.subplot(121)
        plt.imshow(vec1, origin="lower", vmin=vmin1, vmax=vmax1, cmap=cmap1, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel1, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel1)
        plt.ylabel(ylabel1)
        plt.title(title1)

        plt.subplot(122)
        plt.imshow(vec2, origin="lower", vmin=vmin2, vmax=vmax2, cmap=cmap2, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel2, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel2)
        plt.ylabel(ylabel2)
        plt.title(title2)

        if savefile is not "":
            plt.savefig(savefile, bbox_inches="tight")

        if show:
            plt.show()

    def __plot_pad_vec_complex(
            self,
            vec,
            title1="Field",
            title2="Field",
            xlabel1="X",
            ylabel1="Z",
            xlabel2="X",
            ylabel2="Z",
            colorbar=True,
            colorlabel1="",
            colorlabel2="",
            vmin1="",
            vmax1="",
            vmin2="",
            vmax2="",
            cmap1="hot",
            cmap2="hot",
            show=True,
            savefile=""
    ):

        # Plot the velocity field
        if vmin1 is "":
            vmin1 = np.amin(np.real(vec))
        if vmax1 is "":
            vmax1 = np.amax(np.real(vec))

        if vmin2 is "":
            vmin2 = np.amin(np.imag(vec))
        if vmax2 is "":
            vmax2 = np.amax(np.imag(vec))

        # Reshape vec
        vec1 = np.reshape(
            a=np.real(vec),
            newshape=(self.veltrue.geometry2D.gridpointsZ - 2, self.veltrue.geometry2D.gridpointsX - 2)
        )
        vec2 = np.reshape(
            a=np.imag(vec),
            newshape=(self.veltrue.geometry2D.gridpointsZ - 2, self.veltrue.geometry2D.gridpointsX - 2)
        )

        plt.figure()
        plt.subplot(121)
        plt.imshow(vec1, origin="lower", vmin=vmin1, vmax=vmax1, cmap=cmap1, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel1, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel1)
        plt.ylabel(ylabel1)
        plt.title(title1)

        plt.subplot(122)
        plt.imshow(vec2, origin="lower", vmin=vmin2, vmax=vmax2, cmap=cmap2, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel2, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel2)
        plt.ylabel(ylabel2)
        plt.title(title2)

        if savefile is not "":
            plt.savefig(savefile, bbox_inches="tight")

        if show:
            plt.show()

    def __residual_2_modeling_grid(self, vec_residual, vec_out, add_flag=False):

        if not add_flag:
            vec_out = vec_out * 0

        nx_solver = self.veltrue.geometry2D.gridpointsX - 2
        rcvlist = self.veltrue.geometry2D.receivers

        for nreceiver, receiver in enumerate(rcvlist):
            receiver_grid_index = nx_solver * (receiver[1] - 1) + receiver[0] - 1
            vec_out[receiver_grid_index] += vec_residual[nreceiver]

        return vec_out

    def __modeling_grid_2_nopad_grid(self, vec_model_grid, vec_nopad_grid, add_flag=False):

        if not add_flag:
            vec_nopad_grid = vec_nopad_grid * 0

        # Get model grid point info, pad info, nopad grid info
        nx_model = self.veltrue.geometry2D.gridpointsX - 2
        nx_nopad = self.veltrue.geometry2D.ncellsX + 1
        nz_nopad = self.veltrue.geometry2D.ncellsZ + 1
        nx_skip = self.veltrue.geometry2D.ncellsX_pad - 1
        nz_skip = self.veltrue.geometry2D.ncellsZ_pad - 1

        # Copy into cropped field
        for nz in range(nz_nopad):
            start_nopad = nz * nx_nopad
            end_nopad = start_nopad + nx_nopad
            start_model = (nz + nz_skip) * nx_model + nx_skip
            end_model = start_model + nx_nopad

            vec_nopad_grid[start_nopad: end_nopad] += vec_model_grid[start_model: end_model]

        return vec_nopad_grid


if __name__ == "__main__":

    # Define frequency parameters (in Hertz)
    freq_peak_ricker = 10
    freq_max = 20
    flat_spectrum = False
    omega_max = 2 * Common.pi * freq_max
    dt = 0.5 / freq_max
    nt = 100
    domega = (2 * Common.pi) / (nt * dt)
    delay = 0.1

    # Create geometry object
    geom2d = CreateGeometry2D(
        xdim=0.5,
        zdim=0.5,
        vmin=1.5,
        vmax=2.5,
        omega_max=omega_max,
        omega_min=domega
    )
    geom2d.set_params(
        ncells_x=50,
        ncells_z=50,
        ncells_x_pad=75,
        ncells_z_pad=75,
        check=False
    )
    skip_src = 1
    skip_rcv = 1
    geom2d.set_default_sources(skip=skip_src)
    geom2d.set_default_receivers(skip=skip_rcv)

    # Create a default Velocity 2D object
    vel_true = Velocity2D(geometry2d=geom2d)
    ngridpoints_x = geom2d.gridpointsX
    ngridpoints_z = geom2d.gridpointsZ

    # Put Gaussian perturbation in the center
    center_nx = int(ngridpoints_x / 2)
    center_nz = int(ngridpoints_z / 2)
    vel_true.create_gaussian_perturbation(dvel=0.3, sigma_x=0.03, sigma_z=0.03, nx=center_nx, nz=center_nz)

    # Create a Tfwi object, with a constant starting model
    tfwi = Tfwi2D(veltrue=vel_true)
    tfwi.set_constant_starting_model()
    tfwi.veltrue.plot_nopad(
        title="True Model",
        vmin=2.0,
        vmax=2.3,
        xlabel="X grid points",
        ylabel="Z grid points",
        savefile="veltrue.pdf"
    )
    tfwi.velstart.plot_nopad(
        title="Starting Model",
        vmin=2.0,
        vmax=2.3,
        xlabel="X grid points",
        ylabel="Z grid points",
        savefile="velstart.pdf"
    )
    tfwi.veltrue.plot_difference(
        vel_comparison=tfwi.velstart,
        pad=False,
        title="Model Difference",
        xlabel="X grid points",
        ylabel="Z grid points",
        cmap="seismic",
        savefile="veldiff.pdf"
    )

    # omega_list = np.arange(domega, omega_max, domega)
    omega_list = np.arange(omega_max / 8, omega_max + omega_max / 8, omega_max / 8)
    tfwi.set_omega_list(omega_list=omega_list)
    if not flat_spectrum:
        tfwi.set_ricker_wavelet(omega_peak=2.0 * Common.pi * freq_peak_ricker)
    else:
        tfwi.set_flat_spectrum_wavelet()

    tfwi.perform_lsm_cg(
        save_lsm_adjoint_image=True,
        save_lsm_adjoint_allimages=True,
        lsm_adjoint_image_file="lsm-adjoint-image-8"
    )

    # tfwi.perform_lsm_cg_stochastic(
    #     prob=0.6,
    #     save_lsm_adjoint_image=True,
    #     save_lsm_adjoint_allimages=False,
    #     lsm_adjoint_image_file="lsm-adjoint-image-8-p60"
    # )
