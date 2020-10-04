import sys
from pathlib import Path
import pickle
import numpy as np

devito_examples_dir = "/homes/sep/rahul/devito/examples/"
sys.path.append(devito_examples_dir)

# Devito related imports
from devito import Function, TimeFunction
from devito import Operator, Eq, solve, Constant, Inc
from devito.tools import memoized_meth
from examples.seismic import PointSource, TimeAxis, demo_model


class TypeChecker(object):
    """
    A class for all TypeChecker functions needed to support the Fwi2d class
    """

    @staticmethod
    def check(x, expected_type, f=(lambda y: (True, True, ""))):

        str1 = ", ".join([str(types) for types in expected_type])
        if not isinstance(x, expected_type):
            raise TypeError("Object type : " + str(type(x)) + " , Expected types : " + str1)

        type_check, value_check, msg = f(x)

        if not type_check:
            raise TypeError(msg)
        if not value_check:
            raise ValueError(msg)

        return True

    @staticmethod
    def check_int_positive(x):

        str1 = ", ".join([str(types) for types in (int,)])
        if not isinstance(x, int):
            raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

        if x <= 0:
            raise ValueError("Value of x :" + str(x) + " , Expected value : x > 0")

        return True

    @staticmethod
    def check_int_bounds(x, lb, ub):

        str1 = ", ".join([str(types) for types in (int,)])
        if not isinstance(x, int):
            raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

        if not lb <= x <= ub:
            raise ValueError("Value of x :" + str(x) + " , Expected value : " + str(lb) + " <= x <= " + str(ub))

        return True

    @staticmethod
    def check_float_positive(x):

        str1 = ", ".join([str(types) for types in (float, int)])
        if not isinstance(x, (float, int)):
            raise TypeError("Object type : " + str(type(x)) + " , Expected types : " + str1)

        if x <= 0:
            raise ValueError("Value of x :" + str(x) + " , Expected value : x > 0")

        return True


class AcousticSolver(object):
    """
    An acoustic solver class
    """

    def __init__(self, model, dt, nt, time_order, space_order, wavelet):
        """
        model:
            Physical model with domain parameters.
        dt:
            Time sampling interval. (in ms)
        nt:
            Number of time samples in modeling.
        time_order:
            Time order for differentiation.
        space_order:
            Space order for differentiation.
        wavelet:
            Wavelet for modeling (must be 1d numpy array of size nt).
        """

        self.model = model
        self.dt = dt
        self.nt = nt
        self.space_order = space_order
        self.time_order = time_order
        self.wavelet = wavelet
        self.time_axis = TimeAxis(start=0.0, step=dt, num=nt)

        self.forward_op = self.op_fwd(save=False)
        self.forward_op_save = self.op_fwd(save=True)
        self.gradient_op = self.op_grad()
        self.gradient_slsq_op = self.op_grad_slsq()

    @memoized_meth
    def op_fwd(self, save=None):
        """
        Cached operator for forward runs
        """

        # Create symbols for forward wavefield, source, receivers, dt
        u = TimeFunction(
            name='u',
            grid=self.model.grid,
            save=self.nt if save else None,
            time_order=self.time_order,
            space_order=self.space_order
        )
        src = PointSource(name='src', grid=self.model.grid, time_range=self.time_axis, npoint=1)
        rec = PointSource(name='rec', grid=self.model.grid, time_range=self.time_axis, npoint=1)
        dt = Constant(name='dt')

        # Create pde
        pde = self.model.m * u.dt2 - u.laplace + self.model.damp * u.dt
        stencil = Eq(u.forward, solve(pde, u.forward), subdomain=self.model.grid.subdomains['physdomain'])

        # Construct expression to inject source values
        src_term = src.inject(field=u.forward, expr=src * dt ** 2 / self.model.m)

        # Create interpolation expression for receivers
        rec_term = rec.interpolate(expr=u)

        # Substitute spacing terms to reduce flops
        return Operator([stencil] + src_term + rec_term, subs=self.model.spacing_map, name='Forward')

    @memoized_meth
    def op_grad_slsq(self):
        """
        Cached operator for gradient runs
        """

        # Gradient symbol and wavefield symbols, dt symbol
        grad = Function(name='grad', grid=self.model.grid)
        dt = Constant(name='dt')

        # Create symbols for forward wavefield, adjoint wavefield, and receivers
        u = TimeFunction(
            name='u',
            grid=self.model.grid,
            save=self.nt,
            time_order=self.time_order,
            space_order=self.space_order
        )
        v = TimeFunction(
            name='v',
            grid=self.model.grid,
            save=None,
            time_order=self.time_order,
            space_order=self.space_order
        )
        rec = PointSource(name='rec', grid=self.model.grid, time_range=self.time_axis, npoint=1)

        # Create pde and gradient update
        pde = self.model.m * v.dt2 - v.laplace + self.model.damp * v.dt.T
        stencil = Eq(v.backward, solve(pde, v.backward), subdomain=self.model.grid.subdomains['physdomain'])
        gradient_update = Inc(grad, - u.dt2 * v)

        # Add expression for receiver injection
        rec_term = rec.inject(field=v.backward, expr=rec * dt ** 2 / self.model.m)

        # Substitute spacing terms to reduce flops
        return Operator([stencil] + rec_term + [gradient_update], subs=self.model.spacing_map, name='Gradient')

    @memoized_meth
    def op_grad(self):
        """
        Cached operator for gradient runs
        """

        # Gradient symbol and wavefield symbols, dt symbol
        grad = Function(name='grad', grid=self.model.grid)
        dt = Constant(name='dt')

        # Create symbols for forward wavefield, adjoint wavefield, and receivers
        u = TimeFunction(
            name='u',
            grid=self.model.grid,
            save=self.nt,
            time_order=self.time_order,
            space_order=self.space_order
        )
        v = TimeFunction(
            name='v',
            grid=self.model.grid,
            save=None,
            time_order=self.time_order,
            space_order=self.space_order
        )
        rec = PointSource(name='rec', grid=self.model.grid, time_range=self.time_axis, npoint=1)

        # Create pde and gradient update
        pde = self.model.m * v.dt2 - v.laplace + self.model.damp * v.dt.T
        stencil = Eq(v.backward, solve(pde, v.backward), subdomain=self.model.grid.subdomains['physdomain'])
        gradient_update = Inc(grad, u.dt2 * v * (self.model.m ** 1.5))  # factor of 2 ignored

        # Add expression for receiver injection
        rec_term = rec.inject(field=v.backward, expr=rec * dt ** 2 / self.model.m)

        # Substitute spacing terms to reduce flops
        return Operator([stencil] + rec_term + [gradient_update], subs=self.model.spacing_map, name='Gradient')

    def forward(self, src_coords, rec_coords, u=None, save=None):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        @Params
        src_coords:
            Source coordinates (numpy float array of shape 1 x 2)
        rec_coords:
            Source coordinates (numpy float array of shape 1 x nr)
        u:
            Stores the computed wavefield. A TimeFunction object.
        save:
            bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        @Returns
        Receiver, wavefield and performance summary
        """

        # Create a new source object
        src = PointSource(
            name='src',
            grid=self.model.grid,
            time_range=self.time_axis,
            data=self.wavelet,
            coordinates=src_coords,
            space_order=self.space_order,
            time_order=self.time_order
        )

        # Create a new receiver object to store the result
        rec = PointSource(
            name='rec',
            grid=self.model.grid,
            time_range=self.time_axis,
            coordinates=rec_coords,
            space_order=self.space_order,
            time_order=self.time_order
        )

        # Execute operator and return wavefield and receiver data
        if save:
            u = u or TimeFunction(
                name='u',
                grid=self.model.grid,
                save=self.nt,
                time_order=self.time_order,
                space_order=self.space_order
            )
            summary = self.forward_op_save.apply(src=src, rec=rec, u=u, dt=self.dt)
        else:
            u = TimeFunction(
                name='u',
                grid=self.model.grid,
                save=None,
                time_order=self.time_order,
                space_order=self.space_order
            )
            summary = self.forward_op.apply(src=src, rec=rec, u=u, dt=self.dt)

        return rec, u, summary

    def gradient_slsq(self, rec, u, grad=None):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modeling function, ie. the action of the
        Jacobian adjoint on an input data.

        @Params
        rec:
            PointSource (residual). Receiver data.
        u:
            TimeFunction. Full wavefield `u` (created with save=True).
        grad:
            Function, optional. Stores the gradient field.

        @Returns
        Gradient field and performance summary.
        """

        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Compute gradient
        summary = self.gradient_slsq_op.apply(rec=rec, grad=grad, u=u, dt=self.dt)
        return grad, summary

    def gradient(self, rec, u, grad=None):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modeling function, ie. the action of the
        Jacobian adjoint on an input data.

        @Params
        rec:
            PointSource (residual). Receiver data.
        u:
            TimeFunction. Full wavefield `u` (created with save=True).
        grad:
            Function, optional. Stores the gradient field.

        @Returns
        Gradient field and performance summary.
        """

        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Compute gradient
        summary = self.gradient_op.apply(rec=rec, grad=grad, u=u, dt=self.dt)
        return grad, summary


class Fwi2d(object):
    """
    This class has all methods to run a 2D Full Waveform Inversion problem.
    All public methods of this class should ensure that self.__vel stores the current model.
    """

    def __init__(self, model, src_coords, rec_coords, data, params, max_iter, wavelet, **kwargs):
        """
        This function runs fwi on the input arguments

        Note:
            1. No CFL condition checks are done in this function. The user should choose these values carefully
            and set them in the params dictionary input as an argument to this function.

            2. We assume all to all connectivity for sources and receivers

        @Params
        model:
            Starting model (numpy float array of shape Nx x Nz).
        src_coords:
            Source coordinates (numpy float array of shape Ns x 2).
        rec_coords:
            Receiver coordinates (numpy float array of shape Nr x 2).
        data:
            Ns x Nt x Nr (Nt is same as the propagation grid). This is the true recorded data.
        params:
            A dictionary containing all parameters related to the inversion.
        max_iter:
            Maximum number of iterations to run.
        wavelet:
            Numpy float array of shape Nt x 1.
        **kwargs:
            Extra arguments (optional).

        Note: params contains the following keys
            - "Nx" (int): grid points in X direction
            - "Nz" (int): grid points in Z direction
            - "Nt" (int): grid points in time
            - "Ns" (int): number of shots
            - "Nr" (int): number of receivers
            - "Npad" (int): number of grid points for padding
            - "dt" (float): time step (in ms)
            - "dx" (float): grid size in X direction (in meters)
            - "dz" (float): grid size in Z direction (in meters)
            - "so" (int): space order for space derivative computation
            - "to" (int): time order for time derivative computation

        Note: kwargs may currently contain the following keys (other keys will not have any effect)
            - "save_data_model" (bool): Flag to indicate if intermediate modeled data and models will be saved
            - "nsave_data_model" (int): Frequency at which intermediate modeled data and models will be saved
            - "output_dir" (str): Directory where to write all files
            - "stepper" (str): Stepper type for model update
        """

        # -----------------------------------------------------
        # Read in paramaters from params

        self.__nx = params["Nx"]
        self.__nz = params["Nz"]
        self.__nt = params["Nt"]
        self.__ns = params["Ns"]
        self.__nr = params["Nr"]
        self.__npad = params["Npad"]
        self.__npad_total = self.__npad + self.__space_order

        self.__dt = params["dt"]
        self.__dx = params["dx"]
        self.__dz = params["dz"]

        self.__space_order = params["so"]
        self.__time_order = params["to"]

        self.__obj_init = None

        TypeChecker.check_int_positive(x=self.__nx)
        TypeChecker.check_int_positive(x=self.__nz)
        TypeChecker.check_int_positive(x=self.__nt)
        TypeChecker.check_int_positive(x=self.__ns)
        TypeChecker.check_int_positive(x=self.__nr)
        TypeChecker.check_int_positive(x=self.__npad)
        TypeChecker.check_float_positive(x=self.__dt)
        TypeChecker.check_float_positive(x=self.__dx)
        TypeChecker.check_float_positive(x=self.__dz)
        TypeChecker.check_int_bounds(x=self.__space_order, lb=2, ub=10)
        TypeChecker.check_int_bounds(x=self.__time_order, lb=2, ub=4)
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Do some checks on the inputs here and instantiate remaining class members

        self.__starting_model = model
        self.__src_coords = src_coords
        self.__rec_coords = rec_coords
        self.__data = data
        self.__max_iter = max_iter
        self.__wavelet = wavelet
        self.__run_checks_input()
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Read in paramaters from kwargs

        self.__save_data_model = kwargs.pop("save_data_model", False)
        TypeChecker.check(x=self.__save_data_model, expected_type=(bool,))

        if self.__save_data_model:
            self.__nsave_data_model = kwargs.pop("nsave_data_model", 1)
            TypeChecker.check_int_positive(x=self.__nsave_data_model)

        self.__output_dir = kwargs.pop("output_dir", Path().absolute())
        TypeChecker.check(x=self.__output_dir, expected_type=(str,))
        Path(self.__output_dir).mkdir(parents=True, exist_ok=True)

        self.__stepper = kwargs.pop("stepper", "parabolic")
        if self.__stepper not in ["parabolic"]:
            raise ValueError("Only parabolic stepper is currently supported.")
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Pickle the params dictionary and save to disk
        with open(Path(self.__output_dir).joinpath("params.pickle"), "wb") as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Intialize all needed Devito objects

        # Create vel object by padding initial model in all four sides
        model_init_padded = self.__pad_field(field=self.__starting_model)
        self.__vel = self.create_model(
            shape=(self.__nx, self.__nz),
            spacing=(self.__dx, self.__dz),
            npad=self.__npad,
            space_order=self.__space_order
        )
        self.__vel.vp.data_with_halo[:] = model_init_padded

        # Create AcousticSolver object
        self.__solver = AcousticSolver(
            model=self.__vel,
            dt=self.__dt,
            nt=self.__nt,
            time_order=self.__time_order,
            space_order=self.__space_order,
            wavelet=self.__wavelet
        )
        # -----------------------------------------------------

    def calculate_fwi_obj_grad(self, mode=0, slsq=False):
        """
        Define a function that calculates the FWI objective function and optionally calculates the gradient.

        Note:
            This uses the solver object which stores the current model.

        @Params
        mode:
            If 0 then only calculate the objective function, and modeled data.
            If 1 then return gradient and objective function.
        slsq:
            If True, update in slowness squared.
            If False, update in velocity.

        @Returns
        If mode = 0, returns the objective function, and modeled data.
        If mode = 1, returns the gradient, objective function, and modeled data.
        """

        TypeChecker.check_int_bounds(x=mode, lb=0, ub=1)
        TypeChecker.check(x=slsq, expected_type=(bool,))

        # Allocate memory to store modeled data, primary wave field, and gradient
        data_current = self.__data * 0
        u = TimeFunction(
            name='u',
            grid=self.__vel.grid,
            save=self.__nt,
            time_order=self.__time_order,
            space_order=self.__space_order
        )

        # Initialize objective function
        obj = 0.

        if mode == 0:

            # Loop over shots
            for nshot in range(self.__ns):

                # Calculate primary wave field
                rec, u, _ = self.__solver.forward(
                    src_coords=self.__src_coords[nshot, :],
                    rec_coords=self.__rec_coords,
                    u=u,
                    save=True
                )
                data_current[nshot, :, :] = rec.data

                # Calculate residual and update, and update objective function
                rec.data[:] -= self.__data
                obj += np.linalg.norm(rec.data) ** 2

            return obj, data_current

        if mode == 1:
            grad = Function(name='grad', grid=self.__vel.grid)
            grad_total = np.zeros(shape=(self.__nx, self.__nz), dtype=np.float32)

            # Loop over shots
            for nshot in range(self.__ns):

                # Calculate primary wave field
                rec, u, _ = self.__solver.forward(
                    src_coords=self.__src_coords[nshot, :],
                    rec_coords=self.__rec_coords,
                    u=u,
                    save=True
                )
                data_current[nshot, :, :] = rec.data

                # Calculate residual and update, and update objective function
                rec.data[:] -= self.__data
                obj += np.linalg.norm(rec.data) ** 2

                # Calculate gradient and update
                grad.data_with_halo[:] *= 0
                if slsq:
                    grad, _ = self.__solver.gradient_slsq(rec=rec, u=u, grad=grad)
                    grad_total += grad.data
                else:
                    grad, _ = self.__solver.gradient(rec=rec, u=u, grad=grad)
                    grad_total += grad.data

            return grad_total, obj, data_current

    def run(self, slsq, num_iter=None):
        """
        Run Fwi

        @Params
        slsq:
            If True, update in slowness squared.
            If False, update in velocity.
        num_iter:
            Number of iterations to run (optional, mainly for interactive control)
        """

        if num_iter is None:
            num_iter = self.__max_iter
        else:
            TypeChecker.check_int_bounds(x=num_iter, lb=1, ub=self.__max_iter)

        for niter in range(num_iter):

            # Calculate gradient
            grad, obj, data_current = self.calculate_fwi_obj_grad(mode=1, slsq=slsq)
            if niter == 0:
                self.__obj_init = obj

            # Stepper
            last_alpha = 1.0
            if self.__stepper == "parabolic":
                if niter == 0:
                    last_alpha, _, _ = self.parabolic_stepper(grad=grad, phi0=obj, slsq=slsq)
                else:
                    last_alpha, _, _ = self.parabolic_stepper(grad=grad, phi0=obj, alpha0=last_alpha, slsq=slsq)

    # -----------------------------------------------------
    # Line search steppers

    def parabolic_stepper(self, grad, phi0=None, alpha0=None, slsq=False):
        """
        This is the parabolic line search stepper.

        Note:
            No checks on the inputs are done.
            TODO

        @Params
        grad:
            Gradient of Fwi objective function.
        phi0:
            Initial objective function.
        alpha0:
            Initial alpha0 (step length).
        slsq:
            If True, update in slowness squared.
            If False, update in velocity.

        @Returns
        (step_length, updated_model, obj)
        """

        # Save current model (velocity)
        m0 = np.copy(self.__vel.vp.data)

        # Calculate initial objective function if not passed to function
        if phi0 is None:
            phi0, _ = self.calculate_fwi_obj_grad(mode=0, slsq=slsq)

        # Compute initial step length
        if alpha0 is None:
            alpha0 = 1.0 / np.linalg.norm(grad)

        # Perform parabolic line search
        while 1:

            # Set c1, c2
            c1 = 1.0
            c2 = 2.0

            # Compute phi1, phi2
            self.__update_model(model_init=m0, pert=-c1 * alpha0 * grad, slsq=slsq)
            phi1, _ = self.calculate_fwi_obj_grad(mode=0, slsq=slsq)
            self.__update_model(model_init=m0, pert=-c2 * alpha0 * grad, slsq=slsq)
            phi2, _ = self.calculate_fwi_obj_grad(mode=0, slsq=slsq)

            # Fit parabola and compite c_opt
            c_opt = 0.5 * ((c2 ** 2) * (phi1 - phi0) + (c1 ** 2) * (phi0 - phi2)) / \
                (c2 * (phi1 - phi0) + c1 * (phi0 - phi2))

            # Compute phi_opt
            self.__update_model(model_init=m0, pert=-c_opt * alpha0 * grad, slsq=slsq)
            phi_opt, _ = self.calculate_fwi_obj_grad(mode=0, slsq=slsq)

            # Select update
            c = c_opt
            phi = phi_opt
            if phi_opt > phi2:
                c = c2
                phi = phi2
                if phi2 > phi1:
                    c = c1
                    phi = phi1
            else:
                if phi_opt > phi1:
                    c = c1
                    phi = phi1

            alpha = alpha0 * c

            # Check if update smaller than initial objective
            if phi < phi0:
                model_new = self.__update_model(model_init=m0, pert=-alpha * grad, slsq=slsq)
                return alpha, model_new, phi
            else:
                alpha0 = alpha0 * 0.5

    @property
    def vel(self):
        return self.__vel

    @staticmethod
    def create_model(shape=(200, 200), spacing=(10., 10.), npad=75, space_order=4):
        """
        @Params
        shape:
            2d numpy array with shape of model (without padding)

        @Returns
        A Devito velocity model object.
        """
        return demo_model(
            'layers-isotropic',
            origin=(0., 0.),
            shape=shape,
            spacing=spacing,
            nbl=npad,
            space_order=space_order,
            grid=None,
            nlayers=1,
            bcs='damp'
        )

    # -----------------------------------------------------
    # Private methods of the class below here

    def __run_checks_input(self):
        """
        Run checks on types, sizes, and values on some of the remaining class members
        """
        # TODO
        return 0

    def __pad_field(self, field):
        """
        A function that performs padding for fields. The values are copied from last sample in each
        direction to the end of the padded field.

        @Params
        field:
            Numpy float array of shape Nx x Nz

        @Returns
        Padded field.
        """
        field_extend = np.zeros(
            shape=(self.__nx + 2 * self.__npad_total, self.__nz + 2 * self.__npad_total),
            dtype=np.float32
        )
        field_extend[
            self.__npad_total: self.__nx + self.__npad_total,
            self.__npad_total: self.__nz + self.__npad_total
        ] = field

        field_extend[0: self.__npad_total, :] = field_extend[self.__npad_total, :]
        field_extend[self.__nx + self.__npad_total: self.__nx + 2 * self.__npad_total, :] = \
            field_extend[self.__nx + self.__npad_total - 1, :]

        field_extend[:, 0: self.__npad_total] = field_extend[:, self.__npad_total]
        field_extend[:, self.__nz + self.__npad_total: self.__nz + 2 * self.__npad_total] = \
            field_extend[:, self.__nz + self.__npad_total - 1]

        return field_extend

    def __update_model(self, model_init, pert, slsq=False):
        """
        A function that updates the velocity model given a perturbation.

        Note:
            This function modifies the self.__vel class variable.

        @Params
        model_init:
            Numpy float array of shape Nx x Nz (always is velocity)
        pert:
            Numpy float array of shape Nx x Nz (can be velocity or slowness squared)
        slsq:
            If True, update in slowness squared.
            If False, update in velocity.

        @Returns
        Modifies the vel object vp values.
        """

        if not slsq:
            field_padded = self.__pad_field(field=model_init + pert)
            self.__vel.vp.data_with_halo[:] = field_padded

        else:
            slowness_squared = model_init ** (-2.0)
            field_padded = self.__pad_field(field=slowness_squared + pert)
            self.__vel.vp.data_with_halo[:] = field_padded ** (-0.5)

        return self.__vel.vp.data
