# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:04:30 2017
@author: rahul
"""
from Common import*
from CreateGeometry import CreateGeometry2D
from Velocity import Velocity2D
from CreateMatrixHelmholtz import CreateMatrixHelmholtz2D
from Utilities import TypeChecker
import copy
import numpy as np
import time
from scipy.sparse.linalg import splu


class Tfwi2D(object):
    """
    Create a Tfwi problem object for 2D test problems
    """
    def __init__(
            self,
            veltrue=Velocity2D(),
    ):

        ####################################################################################################
        # These quantities cannot be changed after class is initialized
        ####################################################################################################
        self.veltrue = copy.deepcopy(veltrue)
        self.velstart = copy.deepcopy(veltrue)
        self.set_constant_starting_model()

        ####################################################################################################
        # These quantities can be changed after class is initialized
        ####################################################################################################
        self.omega_list = [
            self.veltrue.geometry2D.omega_min,
            self.veltrue.geometry2D.omega_max
        ]
        self.wavelet = []
        self.set_ricker_wavelet(omega_peak=self.veltrue.geometry2D.omega_max / 2.0)

    def set_true_model(self, velocity):

        self.veltrue.vel = velocity

    def set_starting_model(self, velocity):

        self.velstart.vel = velocity

    def set_constant_starting_model(self):

        velavg = (self.veltrue.geometry2D.vmax + self.veltrue.geometry2D.vmin) / 2.0
        self.velstart.set_constant_velocity(velavg)

    def set_omega_list(self, omega_list):

        omega_minimum = self.veltrue.geometry2D.omega_min
        omega_maximum = self.veltrue.geometry2D.omega_max

        if np.all(np.asarray([omega_minimum <= omega <= omega_maximum for omega in omega_list])):
            self.omega_list = omega_list
            self.set_ricker_wavelet(omega_peak=self.veltrue.geometry2D.omega_max / 2.0)
        else:
            raise ValueError("Frequency values outside range supported by geometry grid.")

    def set_ricker_wavelet(self, omega_peak):

        self.wavelet = [
            self.ricker_wavelet_coefficient(
                omega=omega,
                omega_peak=omega_peak
            ) for omega in self.omega_list
        ]

    def set_flat_spectrum_wavelet(self):

        self.wavelet = [1.0 for _ in self.omega_list]

    def ricker_time(self, freq_peak=10.0, nt=250, dt=0.004, delay=0.05):

        t = np.arange(0.0, nt * dt, dt)
        y = (1.0 - 2.0 * (Common.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2)) \
            * np.exp(-(Common.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2))
        return t, y

    def ricker_wavelet_coefficient(self, omega, omega_peak):

        amp = 2.0 * (omega ** 2.0) / (np.sqrt(Common.pi) * (omega_peak ** 3.0)) * np.exp(-(omega / omega_peak) ** 2)
        return np.complex64(amp)

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

    def perform_lsm_cg_stochastic(
            self,
            prob=1.0,
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

                if np.random.uniform(low=0.0, high=1.0) > prob:
                    continue

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
                    cmap1="seismic",
                    cmap2="seismic",
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
                cmap="seismic",
                savefile=lsm_adjoint_image_file + ".pdf"
            )

    """
    # Private methods
    """

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

    def __conjugate_gradients(self, linear_operator, rhs, x0, niter):

        # Calculate initial residual, and residual norm
        r = rhs - linear_operator(x0)
        r_norm = np.linalg.norm(x=r)
        if r_norm < 1e-10:
            return x0
        r_norm_sq = r_norm ** 2

        # Initialize p
        p = copy.deepcopy(r)

        # Initialize residual array, iteration array
        residual = [r_norm]
        iterations = [0]

        # Run CG iterations
        for num_iter in range(niter):

            # Compute A*p and alpha
            Ap = linear_operator(p)
            alpha = r_norm_sq / np.vdot(p, Ap)

            # Update x0, residual
            x0 += alpha * p
            r -= alpha * Ap

            # Calculate beta
            r_norm_new = np.linalg.norm(x=r)
            r_norm_new_sq = r_norm_new ** 2
            beta = r_norm_new_sq / r_norm_sq

            # Check convergence
            if r_norm_new < 1e-10:
                break

            # Update p, residual norm
            p = r + beta * p
            r_norm_sq = r_norm_new_sq

            # Update residual array, iteration array
            residual.append(r_norm_new)
            iterations.append(num_iter)

        return x0, (iterations, residual)


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
