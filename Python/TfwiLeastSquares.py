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
from scipy.sparse.linalg import splu
import numpy as np
import copy
import time
import gc


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
        TypeChecker.check(x=veltrue, expected_type=(Velocity2D,))
        TypeChecker.check(x=velstart, expected_type=(Velocity2D,))
        TypeChecker.check(x=acquisition, expected_type=(Acquisition2D,))

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

        ################################################################################################################
        # Objects which are purely internal
        self.__matfac_velstart = None
        self.__matfac_velstart_h = None
        self.__true_data = None
        self.__residual = None
        self.__epsilon = 0.0
        self.__gamma = 0.0

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

    def set_gaussian_wavelet(self, omega_mean=None, omega_std=None):

        if omega_mean is None:
            omega_mean = self.__geometry2D.omega_max / 2.0
        else:
            TypeChecker.check_float_positive(x=omega_mean)

        if omega_std is None:
            omega_std = self.__geometry2D.omega_max / 4.0
        else:
            TypeChecker.check_float_positive(x=omega_std)

        self.__wavelet = [
            WaveletTools.gaussian_spectrum_coefficient(
                omega=omega,
                omega_mean=omega_mean,
                omega_std=omega_std
            ) for omega in self.__omega_list
        ]
        self.__run_cleanup(flag=6)

    def apply_frequency_taper(self, omega_low, omega_high, omega1, omega2):

        self.__wavelet = [
            WaveletTools.apply_spectrum_taper(
                value=self.__wavelet[index],
                omega=omega,
                omega_low=omega_low,
                omega_high=omega_high,
                omega1=omega1,
                omega2=omega2
            ) for index, omega in enumerate(self.__omega_list)
        ]
        self.__run_cleanup(flag=6)

    def compute_matrix_factorizations(self):

        print("Computing factorized matrices for velstart...")
        helmholtz_velstart = CreateMatrixHelmholtz2D(velocity2d=self.__velstart, pml_damping=Common.pml_damping)

        # Initialize list for pre-computed LU factors
        self.__matfac_velstart = []
        self.__matfac_velstart_h = []

        # Loop over frequencies
        print("Looping over frequencies...\n")
        cum_time_matfac = 0.0

        for nomega, omega in enumerate(self.__omega_list):
            print("Starting computation for nomega = ", nomega, " / ", len(self.__omega_list) - 1)

            # Factorize matrix
            print("Creating factorized matrix...")

            time_fac_start = time.time()
            mat = helmholtz_velstart.create_matrix(omega)
            matfac_lu = splu(mat)
            mat = helmholtz_velstart.create_matrix(omega, conjugate_flag=True)
            matfac_lu_h = splu(mat)
            time_fac_end = time.time()

            print("Matrix factorized. Time for factorization of matrix and its transpose = ",
                  time_fac_end - time_fac_start, " s")
            cum_time_matfac += (time_fac_end - time_fac_start)

            self.__matfac_velstart.append(matfac_lu)
            self.__matfac_velstart_h.append(matfac_lu_h)

        # Print total time for matrix factorization
        print("Matrix factorization for ", len(self.__omega_list), " frequencies took ", cum_time_matfac, "s\n\n")

    def compute_true_data(self):

        self.__true_data = self.__compute_data(velocity2d=self.__veltrue)

    def compute_residual(self):

        if self.__true_data is None:

            print("self.__true_data is None...")
            print("Computing true data")
            self.compute_true_data()
            print("Finished computing true data")

        print("Computing data with starting model")
        self.__residual = self.__compute_data(velocity2d=self.__velstart, matfac=self.__matfac_velstart)
        print("Finished computing data with starting model")

        self.__residual = self.__true_data - self.__residual

    def perform_lsm_cg(
            self,
            epsilon=0.0,
            gamma=0.0,
            init_model=None,
            niter=50,
            save_lsm_image=True,
            save_lsm_allimages=False,
            lsm_image_file="lsm_image",
            lsm_image_data_file="lsm_image_data",
            save_lsm_adjoint_image=True,
            save_lsm_adjoint_allimages=False,
            lsm_adjoint_image_file="lsm_adjoint_image",
            lsm_adjoint_image_data_file="lsm_adjoint_image_data"
    ):

        print("Starting the conjugate gradient least squares migration...\n")

        ####################################################################################################
        # Set epsilon and gamma
        ####################################################################################################
        self.__epsilon = epsilon
        self.__gamma = gamma

        ####################################################################################################
        # Get quantities needed throughout the computation
        ####################################################################################################

        # Get grid point info
        nx_nopad = self.__geometry2D.ncellsX + 1
        nz_nopad = self.__geometry2D.ncellsZ + 1
        num_omega = len(self.__omega_list)

        if num_omega < 2:
            raise AttributeError("Need a minimum of two frequencies for the inversion.")

        ####################################################################################################
        # Pre-compute quantities
        ####################################################################################################

        print("Starting the pre-compute phase...\n")

        if self.__matfac_velstart is None or self.__matfac_velstart_h is None:
            print("self.__matfac_velstart or self.__matfac_velstart_h is None...")
            print("Computing matrix factorizations")
            self.compute_matrix_factorizations()
            print("Finished computing matrix factorizations")

        if self.__true_data is None:
            print("self.__true_data is None...")
            print("Computing true data")
            self.compute_true_data()
            print("Finished computing true data")

        if self.__residual is None:
            print("self.__residual is None...")
            print("Computing residual")
            self.compute_residual()
            print("Finished computing residual")

        print("Computing rhs")
        rhs = self.__compute_rhs_cg()
        print("Finished computing rhs")

        # Plot adjoint images for QC
        if save_lsm_adjoint_allimages:
            for nomega in range(len(self.__omega_list)):
                self.__plot_nopad_vec_complex(
                    vec=rhs[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad],
                    title1="Real",
                    title2="Imag",
                    colorbar=False,
                    show=False,
                    cmap1="Greys",
                    cmap2="Greys",
                    savefile=lsm_adjoint_image_file + "-" + str(nomega) + ".pdf"
                )

            vec1 = np.reshape(
                a=rhs,
                newshape=(len(self.__omega_list), self.__geometry2D.ncellsZ + 1, self.__geometry2D.ncellsX + 1)
            )
            np.savez(lsm_adjoint_image_data_file, vec1)

        # Plot adjoint image
        if save_lsm_adjoint_image:
            lsm_adjoint = np.zeros(shape=(nx_nopad * nz_nopad,), dtype=np.complex64)
            for nomega in range(len(self.__omega_list)):
                lsm_adjoint += rhs[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad]
            self.__plot_nopad_vec_real(
                vec=np.real(lsm_adjoint),
                title="LSM adjoint image",
                colorbar=False,
                show=False,
                cmap="Greys",
                savefile=lsm_adjoint_image_file + ".pdf"
            )

            vec1 = np.reshape(
                a=lsm_adjoint,
                newshape=(self.__geometry2D.ncellsZ + 1, self.__geometry2D.ncellsX + 1)
            )
            np.savez(lsm_adjoint_image_data_file + "_stack", vec1)

        ####################################################################################################
        # Inversion begins here
        ####################################################################################################

        # Set starting solution
        if init_model is None:
            x0 = rhs * 0
        else:
            x0 = copy.deepcopy(x=init_model)

        print("Starting the inversion phase...\n")

        # Perform CG (conjugate gradient)
        x0, metrics = LinearSolvers.conjugate_gradient(
            linear_operator=self.__apply_normal_equation_operator,
            rhs=rhs,
            x0=x0,
            niter=niter
        )

        # Plot inverted images for QC
        if save_lsm_allimages:
            for nomega in range(len(self.__omega_list)):
                self.__plot_nopad_vec_complex(
                    vec=x0[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad],
                    title1="Real",
                    title2="Imag",
                    colorbar=False,
                    show=False,
                    cmap1="Greys",
                    cmap2="Greys",
                    savefile=lsm_image_file + "-" + str(nomega) + ".pdf"
                )

            vec1 = np.reshape(
                a=x0,
                newshape=(len(self.__omega_list), self.__geometry2D.ncellsZ + 1, self.__geometry2D.ncellsX + 1)
            )
            np.savez(lsm_image_data_file, vec1)

        # Plot inverted image
        if save_lsm_image:
            lsm = np.zeros(shape=(nx_nopad * nz_nopad,), dtype=np.complex64)
            for nomega in range(len(self.__omega_list)):
                lsm += x0[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad]
            self.__plot_nopad_vec_real(
                vec=np.real(lsm),
                title="LSM inverted image",
                colorbar=False,
                show=False,
                cmap="Greys",
                savefile=lsm_image_file + ".pdf"
            )

            vec1 = np.reshape(
                a=lsm,
                newshape=(self.__geometry2D.ncellsZ + 1, self.__geometry2D.ncellsX + 1)
            )
            np.savez(lsm_image_data_file + "_stack", vec1)

        return x0, metrics

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

    def __get_shot_receiver_info(self):

        num_src = len(self.__acquisition.sources.keys())
        start = []
        end = []
        total_length = 0

        for i in range(num_src):
            start.append(total_length)
            total_length += len(self.__acquisition.receivers[i])
            end.append(total_length)

        return num_src, start, end, total_length

    def __restriction_operator(self, rcv_indices, b, nx):

        num_rcv = len(rcv_indices)
        sampled_b = np.zeros(shape=(num_rcv,), dtype=np.complex64)

        for i, rcv_index in enumerate(rcv_indices):

            receiver_rel_index_x = rcv_index[0] - 1
            receiver_rel_index_z = rcv_index[1] - 1
            receiver_grid_index = (nx * receiver_rel_index_z) + receiver_rel_index_x

            sampled_b[i] = b[receiver_grid_index]

        return sampled_b

    def __compute_data(self, velocity2d, matfac=None):

        # Get grid point info
        nx_solver = self.__geometry2D.gridpointsX - 2
        nz_solver = self.__geometry2D.gridpointsZ - 2

        # Shot Receiver information needed
        num_src, start_rcv_index, end_rcv_index, num_rcv = self.__get_shot_receiver_info()

        print("Generating true data...\n")
        helmholtz_veltrue = CreateMatrixHelmholtz2D(velocity2d=velocity2d, pml_damping=Common.pml_damping)

        # Allocate space for storing true data
        data = np.zeros(shape=(len(self.__omega_list) * num_rcv,), dtype=np.complex64)

        # Loop over frequencies
        print("Looping over frequencies...\n")
        cum_time_datagen = 0.0

        for nomega, omega in enumerate(self.__omega_list):

            print("Starting computation for nomega = ", nomega, " / ", len(self.__omega_list) - 1)
            time_datagen_start = time.time()

            if matfac is None:
                # Factorize matrix
                print("Creating factorized matrix...")

                time_fac_start = time.time()

                mat = helmholtz_veltrue.create_matrix(omega)
                matfac_lu = splu(mat)

                time_fac_end = time.time()

                print("Matrix factorized. Time for factorization = ", time_fac_end - time_fac_start, " s")

            else:
                matfac_lu = matfac[nomega]

            # Initialize rhs
            b = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)

            # Start cumulative time counter
            cum_time_solve = 0.0

            # Loop over shots
            for nshot in range(num_src):

                # Get relative index of shot wrt Helmholtz grid
                shot_rel_index_x = self.__acquisition.sources[nshot][0] - 1
                shot_rel_index_z = self.__acquisition.sources[nshot][1] - 1
                shot_grid_index = (nx_solver * shot_rel_index_z) + shot_rel_index_x

                time_solve_start = time.time()

                # Create RHS and solve for wave field
                b = b * 0
                b[shot_grid_index] = self.__wavelet[nomega]
                b = matfac_lu.solve(rhs=b)

                # Sample wave field at receivers
                start = nomega * num_rcv + start_rcv_index[nshot]
                end = nomega * num_rcv + end_rcv_index[nshot]
                data[start: end] = self.__restriction_operator(
                    rcv_indices=self.__acquisition.receivers[nshot], b=b, nx=nx_solver
                )

                time_solve_end = time.time()

                print("Solving equation for nshot = ", nshot, " / ", num_src - 1,
                      ", Time to solve = ", time_solve_end - time_solve_start, " s")

                cum_time_solve += (time_solve_end - time_solve_start)

            # Print average time to solution
            print("Solving linear system for ", num_src, "shots took ", cum_time_solve, " s")
            print("Average time for linear solve = ", cum_time_solve / float(num_src), " s")

            time_datagen_end = time.time()
            cum_time_datagen += (time_datagen_end - time_datagen_start)

        # Print total data generation time
        print("Data generation for ", len(self.__omega_list), " frequencies, and ",
              num_src, " shots took ", cum_time_datagen, "s\n\n")

        # Garbage collect
        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)

        return data

    def __compute_rhs_cg(self):

        print("Computing rhs...\n")

        # Get grid point info
        nx_solver = self.__geometry2D.gridpointsX - 2
        nz_solver = self.__geometry2D.gridpointsZ - 2
        nx_nopad = self.__geometry2D.ncellsX + 1
        nz_nopad = self.__geometry2D.ncellsZ + 1

        # Get velocity on modeling grid
        vel_cube = self.__velstart.vel[1:(nx_solver + 1), 1:(nz_solver + 1)] ** 3
        vel_cube = vel_cube.flatten()

        # Shot Receiver information needed
        num_src, start_rcv_index, _, num_rcv = self.__get_shot_receiver_info()

        # Initialize rhs vector
        rhs = np.zeros(shape=(len(self.__omega_list) * nx_nopad * nz_nopad,), dtype=np.complex64)

        # Loop over frequencies
        print("Looping over frequencies...\n")
        cum_time_rhsgen = 0.0

        for nomega in range(len(self.__omega_list)):

            print("Starting rhs computation for nomega = ", nomega, " / ", len(self.__omega_list) - 1)
            time_rhsgen_start = time.time()

            # Initialize rhs, sampled data vector at receivers
            b = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)
            residual = self.__residual[nomega * num_rcv: (nomega + 1) * num_rcv]

            # Start cumulative time counter
            cum_time_solve = 0.0

            # Loop over shots
            for nshot in range(num_src):

                # Get relative index of shot wrt Helmholtz grid
                shot_rel_index_x = self.__acquisition.sources[nshot][0] - 1
                shot_rel_index_z = self.__acquisition.sources[nshot][1] - 1
                shot_grid_index = (nx_solver * shot_rel_index_z) + shot_rel_index_x

                time_solve_start = time.time()

                # Create RHS and solve for primary wave field
                b = b * 0
                b[shot_grid_index] = self.__wavelet[nomega]
                u = self.__matfac_velstart[nomega].solve(rhs=b)

                # Backproject and solve for adjoint wavefield
                b = self.__backproject_residual_2_modeling_grid(
                    rcv_indices=self.__acquisition.receivers[nshot],
                    start_index=start_rcv_index[nshot],
                    residual=residual
                )
                b = self.__matfac_velstart_h[nomega].solve(rhs=b)

                # Multiply with adjoint of primary wavefield
                b = b * np.conjugate(u)

                # Multiply with 2 * omega^2 / c^3
                b = b * (2.0 * self.__omega_list[nomega] * self.__omega_list[nomega])
                b = b / vel_cube

                # Add to rhs
                self.__modeling_grid_2_nopad_grid(
                    vec_model_grid=b,
                    vec_nopad_grid=rhs[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad],
                    add_flag=True
                )

                time_solve_end = time.time()
                cum_time_solve += (time_solve_end - time_solve_start)
                print("Creating rhs for nshot = ", nshot, " / ", num_src - 1,
                      ", Time to solve = ", time_solve_end - time_solve_start, " s")

            # Print average time to solution
            print("Creating rhs for ", num_src, "shots took ", cum_time_solve, " s")
            print("Average time per shot = ", cum_time_solve / float(num_src), " s")

            time_rhsgen_end = time.time()
            cum_time_rhsgen += (time_rhsgen_end - time_rhsgen_start)

        # Print total rhs generation time
        print("Rhs generation for ", len(self.__omega_list), " frequencies, and ",
              num_src, " shots took ", cum_time_rhsgen, "s\n\n")

        # Garbage collect
        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)

        return rhs

    def __modeling_grid_2_nopad_grid(self, vec_model_grid, vec_nopad_grid, add_flag=False):

        if not add_flag:
            vec_nopad_grid *= 0

        # Get model grid point info, pad info, nopad grid info
        nx_model = self.__geometry2D.gridpointsX - 2
        nx_nopad = self.__geometry2D.ncellsX + 1
        nz_nopad = self.__geometry2D.ncellsZ + 1
        nx_skip = self.__geometry2D.ncellsX_pad - 1
        nz_skip = self.__geometry2D.ncellsZ_pad - 1

        # Copy into cropped field
        for nz in range(nz_nopad):
            start_nopad = nz * nx_nopad
            end_nopad = start_nopad + nx_nopad
            start_model = (nz + nz_skip) * nx_model + nx_skip
            end_model = start_model + nx_nopad

            vec_nopad_grid[start_nopad: end_nopad] += vec_model_grid[start_model: end_model]

    def __nopad_grid_2_modeling_grid(self, vec_nopad_grid, vec_model_grid, add_flag=False):

        if not add_flag:
            vec_model_grid *= 0

        # Get model grid point info, pad info, nopad grid info
        nx_model = self.__geometry2D.gridpointsX - 2
        nx_nopad = self.__geometry2D.ncellsX + 1
        nz_nopad = self.__geometry2D.ncellsZ + 1
        nx_skip = self.__geometry2D.ncellsX_pad - 1
        nz_skip = self.__geometry2D.ncellsZ_pad - 1

        # Copy into cropped field
        for nz in range(nz_nopad):
            start_nopad = nz * nx_nopad
            end_nopad = start_nopad + nx_nopad
            start_model = (nz + nz_skip) * nx_model + nx_skip
            end_model = start_model + nx_nopad

            vec_model_grid[start_model: end_model] += vec_nopad_grid[start_nopad: end_nopad]

    def __backproject_residual_2_modeling_grid(self, rcv_indices, start_index, residual):

        # Get grid point info
        nx_solver = self.__geometry2D.gridpointsX - 2
        nz_solver = self.__geometry2D.gridpointsZ - 2

        backprojection = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)

        for i, rcv_index in enumerate(rcv_indices):

            receiver_rel_index_x = rcv_index[0] - 1
            receiver_rel_index_z = rcv_index[1] - 1
            receiver_grid_index = (nx_solver * receiver_rel_index_z) + receiver_rel_index_x

            backprojection[receiver_grid_index] = residual[start_index + i]

        return backprojection

    def __apply_normal_equation_operator(self, vector_in, vector_out, add_flag=False):

        if not add_flag:
            vector_out *= 0

        # Get grid point info
        nx_solver = self.__geometry2D.gridpointsX - 2
        nz_solver = self.__geometry2D.gridpointsZ - 2
        nx_nopad = self.__geometry2D.ncellsX + 1
        nz_nopad = self.__geometry2D.ncellsZ + 1
        num_omega = len(self.__omega_list)

        # Get velocity on modeling grid
        vel_cube = self.__velstart.vel[1:(nx_solver + 1), 1:(nz_solver + 1)] ** 3
        vel_cube = vel_cube.flatten()

        # Shot Receiver information needed
        num_src, start_rcv_index, _, num_rcv = self.__get_shot_receiver_info()

        # Add modeling term
        u = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)
        b = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)
        b1 = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)

        for nomega in range(num_omega):

            # Extract rhs
            self.__nopad_grid_2_modeling_grid(
                vec_nopad_grid=vector_in[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad],
                vec_model_grid=b,
                add_flag=False
            )

            # Multiply with 2 * omega^2 / c^3
            b1 *= 0
            b1 += 2.0 * self.__omega_list[nomega] * self.__omega_list[nomega]
            b1 /= vel_cube

            # Loop over shots
            for nshot in range(num_src):

                # Get relative index of shot wrt Helmholtz grid
                shot_rel_index_x = self.__acquisition.sources[nshot][0] - 1
                shot_rel_index_z = self.__acquisition.sources[nshot][1] - 1
                shot_grid_index = (nx_solver * shot_rel_index_z) + shot_rel_index_x

                # Solve for primary wave field
                u *= 0
                u[shot_grid_index] = self.__wavelet[nomega]
                u = self.__matfac_velstart[nomega].solve(rhs=u)

                # Multiply with b1 * u
                u *= b1
                b *= u

                # Solve the Helmholtz equation
                b = self.__matfac_velstart[nomega].solve(rhs=b)

                # Backproject
                data = self.__restriction_operator(
                    rcv_indices=self.__acquisition.receivers[nshot], b=b, nx=nx_solver
                )
                b *= 0
                for i, rcv_index in enumerate(self.__acquisition.receivers[nshot]):
                    receiver_rel_index_x = rcv_index[0] - 1
                    receiver_rel_index_z = rcv_index[1] - 1
                    receiver_grid_index = (nx_solver * receiver_rel_index_z) + receiver_rel_index_x
                    b[receiver_grid_index] = data[i]

                # Solve the Helmholtz equation (Hermitian conjugate)
                b = self.__matfac_velstart_h[nomega].solve(rhs=b)

                # Multiply with conjugate of primary wavefield
                b *= np.conjugate(u)

                # Add to result
                self.__modeling_grid_2_nopad_grid(
                    vec_model_grid=b,
                    vec_nopad_grid=vector_out[
                                   nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad
                                   ],
                    add_flag=True
                )

        # Garbage collect
        del u, b, b1
        gc.collect()

        # Add regularization term
        f1 = (self.__epsilon ** 2) / (abs(self.__omega_list[1] - self.__omega_list[0]) ** 2)
        f2 = 2 * f1 + self.__gamma ** 2
        f3 = f1 + self.__gamma ** 2

        vector_out[0: nx_nopad * nz_nopad] += \
            f3 * vector_in[0: nx_nopad * nz_nopad] - \
            f1 * vector_in[nx_nopad * nz_nopad: 2 * nx_nopad * nz_nopad]

        for nomega_ in range(1, num_omega - 1):
            vector_out[nomega_ * nx_nopad * nz_nopad: (nomega_ + 1) * nx_nopad * nz_nopad] += \
                f2 * vector_in[nomega_ * nx_nopad * nz_nopad: (nomega_ + 1) * nx_nopad * nz_nopad] - \
                f1 * vector_in[(nomega_ - 1) * nx_nopad * nz_nopad: nomega_ * nx_nopad * nz_nopad] - \
                f1 * vector_in[(nomega_ + 1) * nx_nopad * nz_nopad: (nomega_ + 2) * nx_nopad * nz_nopad]

        vector_out[(num_omega - 1) * nx_nopad * nz_nopad: num_omega * nx_nopad * nz_nopad] += \
            f3 * vector_in[(num_omega - 1) * nx_nopad * nz_nopad: num_omega * nx_nopad * nz_nopad] - \
            f1 * vector_in[(num_omega - 2) * nx_nopad * nz_nopad: (num_omega - 1) * nx_nopad * nz_nopad]

    def __apply_forward(self, vector_in, vector_out, add_flag=False):

        if not add_flag:
            vector_out *= 0

        # Get grid point info
        nx_solver = self.__geometry2D.gridpointsX - 2
        nz_solver = self.__geometry2D.gridpointsZ - 2
        nx_nopad = self.__geometry2D.ncellsX + 1
        nz_nopad = self.__geometry2D.ncellsZ + 1

        # Shot Receiver information
        num_src, start_rcv_index, end_rcv_index, num_rcv = self.__get_shot_receiver_info()

        # Get velocity on modeling grid
        vel_cube = self.__velstart.vel[1:(nx_solver + 1), 1:(nz_solver + 1)] ** 3
        vel_cube = vel_cube.flatten()

        # Initialize vectors
        u = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)
        b = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)

        for nomega, omega in enumerate(self.__omega_list):

            # Extract extended model
            self.__nopad_grid_2_modeling_grid(
                vec_nopad_grid=vector_in[nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad],
                vec_model_grid=b,
                add_flag=False
            )

            # Multiply with 2 * omega^2 / c^3
            b *= 2.0 * self.__omega_list[nomega] * self.__omega_list[nomega]
            b /= vel_cube

            # Loop over shots
            for nshot in range(num_src):

                # Get relative index of shot wrt Helmholtz grid
                shot_rel_index_x = self.__acquisition.sources[nshot][0] - 1
                shot_rel_index_z = self.__acquisition.sources[nshot][1] - 1
                shot_grid_index = (nx_solver * shot_rel_index_z) + shot_rel_index_x

                # Solve for primary wave field
                u *= 0
                u[shot_grid_index] = self.__wavelet[nomega]
                u = self.__matfac_velstart[nomega].solve(rhs=u)

                # Multiply with b
                u *= b

                # Solve the Helmholtz equation
                u = self.__matfac_velstart[nomega].solve(rhs=u)

                # Sample wave field at receivers
                start = nomega * num_rcv + start_rcv_index[nshot]
                end = nomega * num_rcv + end_rcv_index[nshot]
                vector_out[start: end] += self.__restriction_operator(
                    rcv_indices=self.__acquisition.receivers[nshot], b=u, nx=nx_solver
                )

        # Garbage collect
        gc.collect()

    def __apply_adjoint(self, vector_in, vector_out, add_flag=False):

        if not add_flag:
            vector_out *= 0

        # Get grid point info
        nx_solver = self.__geometry2D.gridpointsX - 2
        nz_solver = self.__geometry2D.gridpointsZ - 2
        nx_nopad = self.__geometry2D.ncellsX + 1
        nz_nopad = self.__geometry2D.ncellsZ + 1

        # Shot Receiver information
        num_src, start_rcv_index, end_rcv_index, num_rcv = self.__get_shot_receiver_info()

        # Get velocity on modeling grid
        vel_cube = self.__velstart.vel[1:(nx_solver + 1), 1:(nz_solver + 1)] ** 3
        vel_cube = vel_cube.flatten()

        # Initialize vectors
        u = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)
        b = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)
        b1 = np.zeros(shape=(nx_solver * nz_solver,), dtype=np.complex64)

        for nomega, omega in enumerate(self.__omega_list):

            # Multiply with 2 * omega^2 / c^3
            b *= 0
            b += 2.0 * self.__omega_list[nomega] * self.__omega_list[nomega]
            b /= vel_cube

            # Loop over shots
            for nshot in range(num_src):

                # Get relative index of shot wrt Helmholtz grid
                shot_rel_index_x = self.__acquisition.sources[nshot][0] - 1
                shot_rel_index_z = self.__acquisition.sources[nshot][1] - 1
                shot_grid_index = (nx_solver * shot_rel_index_z) + shot_rel_index_x

                # Solve for primary wave field
                u *= 0
                u[shot_grid_index] = self.__wavelet[nomega]
                u = self.__matfac_velstart[nomega].solve(rhs=u)

                u = b * np.conjugate(u)

                # Backproject
                start = nomega * num_rcv + start_rcv_index[nshot]
                end = nomega * num_rcv + end_rcv_index[nshot]
                data = vector_in[start: end]

                b1 *= 0
                for i, rcv_index in enumerate(self.__acquisition.receivers[nshot]):
                    receiver_rel_index_x = rcv_index[0] - 1
                    receiver_rel_index_z = rcv_index[1] - 1
                    receiver_grid_index = (nx_solver * receiver_rel_index_z) + receiver_rel_index_x
                    b1[receiver_grid_index] = data[i]

                # Solve the Helmholtz equation (Hermitian conjugate)
                b1 = self.__matfac_velstart_h[nomega].solve(rhs=b1)

                # Multiply by u
                b1 *= u

                # Add to result
                self.__modeling_grid_2_nopad_grid(
                    vec_model_grid=b1,
                    vec_nopad_grid=vector_out[
                                   nomega * nx_nopad * nz_nopad: (nomega + 1) * nx_nopad * nz_nopad
                                   ],
                    add_flag=True
                )

        # Garbage collect
        gc.collect()

    """
    # Methods below need to be checked thoroughly
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
            self.__run_default_cleanup()

        if flag == 2:
            """
            If self.__veltrue is reset
            """
            self.__run_default_cleanup()

        if flag == 3:
            """
            If self.__velstart is reset
            """
            self.__run_default_cleanup()

        if flag == 4:
            """
            If self.__acquisition is reset
            """
            self.__run_default_cleanup()

        if flag == 5:
            """
            If self.__omega_list is reset
            """
            self.set_ricker_wavelet(omega_peak=self.__geometry2D.omega_max / 2.0)
            self.__run_default_cleanup()

        if flag == 6:
            """
            If self.__wavelet is reset
            """
            self.__run_default_cleanup()

    def __run_default_cleanup(self):

        try:
            del self.__true_data
        except AttributeError:
            print("self.__true_data already deleted.")

        try:
            del self.__residual
        except AttributeError:
            print("self.__residual already deleted.")

        try:
            del self.__matfac_velstart
        except AttributeError:
            print("self.__matfac_velstart already deleted.")

        try:
            del self.__matfac_velstart_h
        except AttributeError:
            print("self.__matfac_velstart_h already deleted.")

        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)

        self.__matfac_velstart = None
        self.__matfac_velstart_h = None
        self.__true_data = None
        self.__residual = None
        self.__epsilon = 0.0
        self.__gamma = 0.0

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
        vec1 = np.reshape(a=vec, newshape=(self.__geometry2D.ncellsZ + 1, self.__geometry2D.ncellsX + 1))

        plt.figure()
        plt.imshow(vec1, origin="upper", vmin=vmin, vmax=vmax, cmap=cmap, interpolation="bilinear")
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
            newshape=(self.__geometry2D.gridpointsZ - 2, self.__geometry2D.gridpointsX - 2)
        )

        plt.figure()
        plt.imshow(vec1, origin="upper", vmin=vmin, vmax=vmax, cmap=cmap, interpolation="bilinear")
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
            newshape=(self.__geometry2D.ncellsZ + 1, self.__geometry2D.ncellsX + 1)
        )
        vec2 = np.reshape(
            a=np.imag(vec),
            newshape=(self.__geometry2D.ncellsZ + 1, self.__geometry2D.ncellsX + 1)
        )

        plt.figure()
        plt.subplot(121)
        plt.imshow(vec1, origin="upper", vmin=vmin1, vmax=vmax1, cmap=cmap1, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel1, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel1)
        plt.ylabel(ylabel1)
        plt.title(title1)

        plt.subplot(122)
        plt.imshow(vec2, origin="upper", vmin=vmin2, vmax=vmax2, cmap=cmap2, interpolation="bilinear")
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
            newshape=(self.__geometry2D.gridpointsZ - 2, self.__geometry2D.gridpointsX - 2)
        )
        vec2 = np.reshape(
            a=np.imag(vec),
            newshape=(self.__geometry2D.gridpointsZ - 2, self.__geometry2D.gridpointsX - 2)
        )

        plt.figure()
        plt.subplot(121)
        plt.imshow(vec1, origin="upper", vmin=vmin1, vmax=vmax1, cmap=cmap1, interpolation="bilinear")
        if colorbar:
            cb = plt.colorbar()
            cb.set_label(colorlabel1, labelpad=-40, y=1.05, rotation=0)
        plt.xlabel(xlabel1)
        plt.ylabel(ylabel1)
        plt.title(title1)

        plt.subplot(122)
        plt.imshow(vec2, origin="upper", vmin=vmin2, vmax=vmax2, cmap=cmap2, interpolation="bilinear")
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


if __name__ == "__main__":

    # Define frequency parameters (in Hertz)
    freq_peak_ricker = 25
    freq_max = 35
    flat_spectrum = False
    omega_max = 2 * Common.pi * freq_max
    omega_min = 2 * Common.pi * freq_peak_ricker / 2.0
    dt = 0.5 / freq_max
    nt = 100
    domega = (2 * Common.pi) / (nt * dt)
    delay = 0.1

    # Create geometry object
    geom2d = CreateGeometry2D(
        xdim=3.0,
        zdim=2.0,
        vmin=1.5,
        vmax=2.5,
        omega_max=omega_max,
        omega_min=omega_min
    )
    geom2d.set_default_params()

    print("Number of grid points in X", geom2d.gridpointsX)
    print("Number of grid points in Z", geom2d.gridpointsZ)
    print("Number of cells in X", geom2d.ncellsX)
    print("Number of cells in Z", geom2d.ncellsZ)
    print("Number of pad cells in X", geom2d.ncellsX_pad)
    print("Number of pad cells in Z", geom2d.ncellsZ_pad)

    # Create acquisition object
    skip_src = 10
    skip_rcv = 1
    acq2d = Acquisition2D(geometry2d=geom2d)
    acq2d.set_split_spread_acquisition(source_skip=skip_src, receiver_skip=skip_rcv, max_offset=1.5)

    # Create a default Velocity 2D object
    vel_true = Velocity2D(geometry2d=geom2d)
    vel_start = Velocity2D(geometry2d=geom2d)
    ngridpoints_x = geom2d.gridpointsX
    ngridpoints_z = geom2d.gridpointsZ

    # Put Gaussian perturbation
    sigma_x_gaussian = 0.2
    sigma_z_gaussian = 0.2
    center_nx = int(ngridpoints_x / 2)
    center_nz = int(ngridpoints_z / 2.5)

    vel_true.set_constant_velocity(vel=2.3)
    vel_true.create_gaussian_perturbation(
        dvel=-0.6,
        sigma_x=sigma_x_gaussian,
        sigma_z=sigma_z_gaussian,
        nx=center_nx,
        nz=center_nz
    )
    vel = vel_true.vel
    vel[:, center_nz + 195: center_nz + 205] = 2.0
    vel_true.vel = vel

    vel_start.set_constant_velocity(vel=2.3)
    vel_start.create_gaussian_perturbation(
        dvel=-0.6,
        sigma_x=sigma_x_gaussian,
        sigma_z=sigma_z_gaussian,
        nx=center_nx,
        nz=center_nz
    )

    # Crop acquisition
    acq2d.crop_sources_receivers_bounding_box(
        nx_start=center_nx - 25,
        nx_end=center_nx + 25,
        nz_start=geom2d.ncellsZ_pad,
        nz_end=geom2d.ncellsZ_pad + geom2d.ncellsZ
    )

    # Create a Tfwi object
    tfwilsq = TfwiLeastSquares2D(veltrue=vel_true, velstart=vel_start, acquisition=acq2d)

    tfwilsq.veltrue.plot(
        title="True Model",
        pad=False,
        vmin=1.5,
        vmax=2.3,
        xlabel="X grid points",
        ylabel="Z grid points",
        savefile="Fig/veltrue-anomaly-bigmodel.pdf"
    )
    tfwilsq.velstart.plot(
        title="Starting Model",
        pad=False,
        vmin=1.5,
        vmax=2.3,
        xlabel="X grid points",
        ylabel="Z grid points",
        savefile="Fig/velstart-anomaly-bigmodel.pdf"
    )
    tfwilsq.veltrue.plot_difference(
        vel_other=tfwilsq.velstart,
        pad=False,
        title="Model Difference",
        xlabel="X grid points",
        ylabel="Z grid points",
        vmin=-0.5,
        vmax=0.5,
        cmap="Greys",
        savefile="Fig/veldiff-anomaly-bigmodel.pdf"
    )

    omega_list = np.arange(domega, omega_max, domega)
    omega_list = np.arange(omega_min, omega_max, (omega_max - omega_min) / 40.0).tolist()
    tfwilsq.omega_list = omega_list
    if not flat_spectrum:
        tfwilsq.set_ricker_wavelet(omega_peak=2.0 * Common.pi * freq_peak_ricker)
    else:
        tfwilsq.set_flat_spectrum_wavelet()

    inverted_model, inversion_metrics = tfwilsq.perform_lsm_cg(
        epsilon=0,
        gamma=0,
        niter=40,
        save_lsm_image=True,
        save_lsm_allimages=True,
        lsm_image_file="Fig/lsm-image-anomaly-acqhole1-bigmodel-eps0",
        save_lsm_adjoint_image=True,
        save_lsm_adjoint_allimages=False,
        lsm_adjoint_image_file="Fig/lsm-adjoint-image-anomaly-acqhole1-bigmodel"
    )

    print(inversion_metrics)
