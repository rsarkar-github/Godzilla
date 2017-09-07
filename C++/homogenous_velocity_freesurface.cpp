#include <iostream>
#include <stdio.h>
#include "easy_io.h"
#include "velocity2D.h"
#include "field2D.h"
#include "sparse_direct_solver2D.h"
#include "boundary_condition2D.h"
#include "umfpack.h"

int main() {

	size_t ncellsX = 500, ncellsY = 500;
	double startX = 0., startY = 0.;
	double endX = 10., endY = 10.;

	Godzilla::Geometry2D geom2D(startX, endX, ncellsX, startY, endY, ncellsY, "x", "y");

	Godzilla::xd scalar(1., 0.);
	Godzilla::vecd data(geom2D.get_nX() * geom2D.get_nY(), 0.);
	data[(geom2D.get_nX() * geom2D.get_nY()) / 2] = 1.;

	Godzilla::Velocity2D vel2D(geom2D, scalar);
	Godzilla::Field2D forcing2D(geom2D, data);
	Godzilla::Field2D solution2D(geom2D);
	Godzilla::BoundaryCondition2D bc2D(geom2D, "PML", "NBC", "PML", "PML");
	double omega = 40;

	Godzilla::Helmholtz2DReal::SparseDirectSolver2D solver(&vel2D, &forcing2D, &bc2D, omega, 0);
	solver.create_sparse_matrix_rhs();
	solver.solve();
	solver.extract_solution(solution2D);

	Godzilla::vecd solution2Dreal, solution2Dimag;
	size_t nelem = solution2D.get_nelem();
	solution2Dreal.assign(nelem, 0.);
	solution2Dimag.assign(nelem, 0.);

	const Godzilla::xd *ptr_sol1 = solution2D.get_cdata().data();
	double *ptr_sol_real = solution2Dreal.data();
	double *ptr_sol_imag = solution2Dimag.data();
	for (size_t i = 0; i < nelem; ++i) {
		ptr_sol_real[i] = ptr_sol1[i].real();
		ptr_sol_imag[i] = ptr_sol1[i].imag();
	}

	wavemod2d::EasyIO io;
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/real_homog_freesurface.bin", solution2Dreal);
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/imag_homog_freesurface.bin", solution2Dimag);

	return 0;
}
