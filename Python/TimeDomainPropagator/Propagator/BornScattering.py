import numpy as np
import scipy.ndimage as spim
import math


def ricker_time(freq_peak=10.0, nt=250, dt=0.004, delay=0.05):

    t = np.arange(0.0, nt * dt, dt, dtype=np.float32)
    y = (1.0 - 2.0 * (math.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2)) \
        * np.exp(-(math.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2))
    return t, y


def cosine_taper_2d(array2d, ncells_pad_x: int, ncells_pad_z: int, get_mask_only=False):

    # Get grid information
    grid_points_z, grid_points_x = array2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
        raise ValueError

    # Create a filter array
    filter_x = np.zeros((1, grid_points_x), dtype=np.float32) + 1.0
    filter_z = np.zeros((grid_points_z, 1), dtype=np.float32) + 1.0

    # Update filter array to be sine square taper
    if ncells_pad_x > 0:
        t = np.zeros((1, ncells_pad_x), dtype=np.float32)
        for i in range(0, ncells_pad_x):
            t[0, i] = i
        t = np.sin(t * (math.pi * 0.5 / ncells_pad_x)) ** 2.0
        filter_x[0, 0:ncells_pad_x] = t[0, :]
        filter_x[0, grid_cells_x - ncells_pad_x + 1:grid_points_x] = t[0, ::-1]

    if ncells_pad_z > 0:
        t = np.zeros((ncells_pad_z, 1), dtype=np.float32)
        for i in range(0, ncells_pad_z):
            t[i, 0] = i
        t = np.sin(t * (math.pi * 0.5 / ncells_pad_z)) ** 2.0
        filter_z[0:ncells_pad_z, 0] = t[:, 0]
        filter_z[grid_cells_z - ncells_pad_z + 1:grid_points_z, 0] = t[::-1, 0]

    if not get_mask_only:
        # Apply filter
        array2d *= filter_x
        array2d *= filter_z
        return

    else:
        # Get filter
        t = np.zeros((grid_points_z, grid_points_x), dtype=np.float32) + 1.0
        t *= filter_x
        t *= filter_z
        return t


def boxcar_taper_2d(array2d, ncells_pad_x: int, ncells_pad_z: int, get_mask_only=False):

    # Get grid information
    grid_points_z, grid_points_x = array2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
        raise ValueError

    # Create a filter array
    filter_x = np.zeros((1, grid_points_x), dtype=np.float32) + 1.0
    filter_z = np.zeros((grid_points_z, 1), dtype=np.float32) + 1.0

    # Update filter array to be sine square taper
    if ncells_pad_x > 0:
        t = np.zeros((1, ncells_pad_x), dtype=np.float32)
        filter_x[0, 0:ncells_pad_x] = t[0, :]
        filter_x[0, grid_cells_x - ncells_pad_x + 1:grid_points_x] = t[0, ::-1]

    if ncells_pad_z > 0:
        t = np.zeros((ncells_pad_z, 1), dtype=np.float32)
        filter_z[0:ncells_pad_z, 0] = t[:, 0]
        filter_z[grid_cells_z - ncells_pad_z + 1:grid_points_z, 0] = t[::-1, 0]

    if not get_mask_only:
        # Apply filter
        array2d *= filter_x
        array2d *= filter_z
        return

    else:
        # Get filter
        t = np.zeros((grid_points_z, grid_points_x), dtype=np.float32) + 1.0
        t *= filter_x
        t *= filter_z
        return t


def laplacian(array2d_in, array2d_out, dx: float, dz: float, order=10):

    if array2d_in.shape != array2d_out.shape:
        raise ValueError

    if order not in [10]:
        raise NotImplementedError

    # Compute 2d stencil
    stencil = []
    if order == 10:
        stencil = [
            0.00031746,
            -0.00496031,
            0.0396825,
            -0.238095,
            1.66667,
            -2.92722,
            1.66667,
            -0.238095,
            0.0396825,
            -0.00496031,
            0.00031746
        ]

    stencil = np.asarray(stencil, dtype=np.float32)
    stencil2d_size = stencil.shape[0]
    stencil2d = np.zeros((stencil2d_size, stencil2d_size), dtype=np.float32)
    stencil2d[int(stencil2d_size / 2), :] += stencil / (dx ** 2.0)
    stencil2d[:, int(stencil2d_size / 2)] += stencil / (dz ** 2.0)

    # Compute Laplacian
    spim.convolve(input=array2d_in, weights=stencil2d, output=array2d_out, mode='nearest')


def acoustic_propagator(
        vel2d, dx: float, dz: float, dt: float, fmax: float,
        source_wavefield,
        propagated_wavefield,
        ncells_pad_x: int=0,
        ncells_pad_z: int=0,
        check_params=True
):

    if check_params:
        # Check dimensions
        if source_wavefield.shape != propagated_wavefield.shape:
            raise ValueError
        if source_wavefield[0, :, :].shape != vel2d.shape:
            raise ValueError

    # Get grid information
    grid_points_z, grid_points_x = vel2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if check_params:
        if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
            raise ValueError

    time_steps = source_wavefield.shape[0]

    if check_params:
        # Check CFL conditions
        vmin = np.min(vel2d)
        vmax = np.max(vel2d)
        f = vmax * dt * np.sqrt(2.0) / np.min([dx, dz])
        if f >= 1:
            raise ValueError("CFL conditions violated for 2d propagation")

        # Check numerical dispersion condition
        f = vmin / (fmax * np.max([dx, dz]))
        if f <= 8.0:
            raise ValueError("Numerical dispersion conditions violated")

    # Get taper filter mask
    taper_mask = cosine_taper_2d(
        array2d=source_wavefield[0, :, :],
        ncells_pad_x=ncells_pad_x,
        ncells_pad_z=ncells_pad_z,
        get_mask_only=True
    )

    # Start propagation
    propagated_wavefield *= 0
    f1 = (vel2d * dt) ** 2.0

    if time_steps >= 2:
        u_next = propagated_wavefield[1, :, :]
        u_next += source_wavefield[0, :, :]
        u_next *= f1
        u_next *= taper_mask

    for i in range(2, time_steps):

        u_prev = propagated_wavefield[i - 2, :, :]
        u_curr = propagated_wavefield[i - 1, :, :]
        u_next = propagated_wavefield[i, :, :]

        laplacian(
            array2d_in=u_curr,
            array2d_out=u_next,
            dx=dx,
            dz=dz,
            order=10
        )
        u_next += source_wavefield[i - 1, :, :]
        u_next *= f1
        u_next += 2 * u_curr - u_prev
        u_next *= taper_mask
        u_curr *= taper_mask


def born_time_dependent_pert_propagator(
        vel2d, dx: float, dz: float, dt: float, fmax: float,
        vel_pert2d,
        source_wavefield,
        born_scattered_wavefield,
        ncells_pad_x: int=0,
        ncells_pad_z: int=0,
        check_params=True,
        adjoint_mode=False
):
    if check_params:
        # Check dimensions
        if source_wavefield.shape != born_scattered_wavefield.shape or source_wavefield.shape != vel_pert2d.shape:
            raise ValueError
        if source_wavefield[0, :, :].shape != vel2d.shape:
            raise ValueError

    # Get grid information
    grid_points_z, grid_points_x = vel2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if check_params:
        if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
            raise ValueError

    time_steps = source_wavefield.shape[0]

    if check_params:
        # Check CFL conditions
        vmin = np.min(vel2d)
        vmax = np.max(vel2d)
        f = vmax * dt * np.sqrt(2.0) / np.min([dx, dz])
        if f >= 1:
            raise ValueError("CFL conditions violated for 2d propagation")

        # Check numerical dispersion condition
        f = vmin / (fmax * np.max([dx, dz]))
        if f <= 8.0:
            raise ValueError("Numerical dispersion conditions violated")

    # Get boxcar filter mask
    boxcar_mask = boxcar_taper_2d(
        array2d=source_wavefield[0, :, :],
        ncells_pad_x=ncells_pad_x,
        ncells_pad_z=ncells_pad_z,
        get_mask_only=True
    )

    # Apply boxcar mask to velocity pertubation, source wavefield, and born scattered wavefield
    vel_pert2d *= boxcar_mask
    source_wavefield *= boxcar_mask
    born_scattered_wavefield *= boxcar_mask

    # Compute primary wavefield
    primary_wavefield = np.zeros((time_steps, grid_points_z, grid_points_x), dtype=np.float32)

    acoustic_propagator(
        vel2d=vel2d,
        dx=dx, dz=dz, dt=dt, fmax=fmax,
        source_wavefield=source_wavefield,
        propagated_wavefield=primary_wavefield,
        ncells_pad_z=ncells_pad_z,
        ncells_pad_x=ncells_pad_x,
        check_params=False
    )

    # Compute 2nd derivative
    laplacian_filter = np.asarray([1, -2, 1], dtype=np.float32) / (dt ** 2.0)
    primary_wavefield = spim.convolve1d(
        input=primary_wavefield,
        weights=laplacian_filter,
        axis=0,
        mode='constant',
        cval=0.0
    )

    # Compute vel cubed inverse
    velcube_inv = vel2d ** (-3.0)

    if not adjoint_mode:
        # Compute secondary source
        primary_wavefield *= np.reshape(2 * velcube_inv, newshape=(1, grid_points_z, grid_points_x)) * vel_pert2d

        # Propagate secondary source
        acoustic_propagator(
            vel2d=vel2d,
            dx=dx, dz=dz, dt=dt, fmax=fmax,
            source_wavefield=primary_wavefield,
            propagated_wavefield=born_scattered_wavefield,
            ncells_pad_z=ncells_pad_z,
            ncells_pad_x=ncells_pad_x,
            check_params=False
        )

    else:
        # Compute diagonal operator
        primary_wavefield *= np.reshape(2 * velcube_inv * boxcar_mask, newshape=(1, grid_points_z, grid_points_x))

        # Propagate born scattered wavefield as source
        acoustic_propagator(
            vel2d=vel2d,
            dx=dx, dz=dz, dt=dt, fmax=fmax,
            source_wavefield=np.flip(born_scattered_wavefield, axis=0),
            propagated_wavefield=np.flip(vel_pert2d, axis=0),
            ncells_pad_z=ncells_pad_z,
            ncells_pad_x=ncells_pad_x,
            check_params=False
        )

        # Multiply with diagonal operator
        vel_pert2d *= primary_wavefield


def born_time_dependent_pert_normal_op(
        vel2d, dx: float, dz: float, dt: float, fmax: float,
        vel_pert2d,
        output,
        source_wavefield,
        restriction_mask,
        ncells_pad_x: int=0,
        ncells_pad_z: int=0,
        check_params=True,
        precomputed_primary_wavefield=False
):
    if check_params:
        # Check dimensions
        if source_wavefield.shape != vel_pert2d.shape or output.shape != vel_pert2d.shape:
            raise ValueError
        if source_wavefield[0, :, :].shape != vel2d.shape or vel2d.shape != restriction_mask.shape:
            raise ValueError

    # Get grid information
    grid_points_z, grid_points_x = vel2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if check_params:
        if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
            raise ValueError

    time_steps = source_wavefield.shape[0]

    if check_params:
        # Check CFL conditions
        vmin = np.min(vel2d)
        vmax = np.max(vel2d)
        f = vmax * dt * np.sqrt(2.0) / np.min([dx, dz])
        if f >= 1:
            raise ValueError("CFL conditions violated for 2d propagation")

        # Check numerical dispersion condition
        f = vmin / (fmax * np.max([dx, dz]))
        if f <= 8.0:
            raise ValueError("Numerical dispersion conditions violated")

    # Get boxcar filter mask
    boxcar_mask = boxcar_taper_2d(
        array2d=source_wavefield[0, :, :],
        ncells_pad_x=ncells_pad_x,
        ncells_pad_z=ncells_pad_z,
        get_mask_only=True
    )

    # Apply boxcar mask to velocity perturbation, source wavefield
    vel_pert2d *= boxcar_mask
    source_wavefield *= boxcar_mask

    # Allocate memory
    primary_wavefield = np.zeros((time_steps, grid_points_z, grid_points_x), dtype=np.float32)
    born_scattered_wavefield = np.zeros((time_steps, grid_points_z, grid_points_x), dtype=np.float32)

    if not precomputed_primary_wavefield:
        # Compute primary wavefield
        acoustic_propagator(
            vel2d=vel2d,
            dx=dx, dz=dz, dt=dt, fmax=fmax,
            source_wavefield=source_wavefield,
            propagated_wavefield=primary_wavefield,
            ncells_pad_z=ncells_pad_z,
            ncells_pad_x=ncells_pad_x,
            check_params=False
        )

    else:
        primary_wavefield += source_wavefield

    # Compute 2nd derivative
    laplacian_filter = np.asarray([1, -2, 1], dtype=np.float32) / (dt ** 2.0)
    primary_wavefield = spim.convolve1d(
        input=primary_wavefield,
        weights=laplacian_filter,
        axis=0,
        mode='constant',
        cval=0.0
    )

    # Compute vel cubed inverse
    velcube_inv = vel2d ** (-3.0)

    # Compute diagonal operator
    primary_wavefield *= np.reshape(2 * velcube_inv * boxcar_mask, newshape=(1, grid_points_z, grid_points_x))

    # Forward
    acoustic_propagator(
        vel2d=vel2d,
        dx=dx, dz=dz, dt=dt, fmax=fmax,
        source_wavefield=primary_wavefield * vel_pert2d,
        propagated_wavefield=born_scattered_wavefield,
        ncells_pad_z=ncells_pad_z,
        ncells_pad_x=ncells_pad_x,
        check_params=False
    )
    born_scattered_wavefield *= boxcar_mask * restriction_mask

    # Adjoint
    acoustic_propagator(
        vel2d=vel2d,
        dx=dx, dz=dz, dt=dt, fmax=fmax,
        source_wavefield=np.flip(born_scattered_wavefield, axis=0),
        propagated_wavefield=np.flip(output, axis=0),
        ncells_pad_z=ncells_pad_z,
        ncells_pad_x=ncells_pad_x,
        check_params=False
    )
    output *= primary_wavefield
