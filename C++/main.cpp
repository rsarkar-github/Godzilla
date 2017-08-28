#include <iostream>
#include <stdio.h>
#include "velocity2D.h"
#include "field2D.h"
#include "sparse_direct_solver2D.h"
#include "boundary_condition2D.h"
#include "umfpack.h"

int main() {

	size_t ncellsX = 6, ncellsY = 6;
	double startX = 0., startY = 0.;
	double endX = 1., endY = 1.;

	Godzilla::Geometry2D geom2D(startX, endX, ncellsX, startY, endY, ncellsY, "x", "y");

	Godzilla::xd scalar(1., 0.);
	Godzilla::vecd data(geom2D.get_nX() * geom2D.get_nY(), 0.);
	data[(geom2D.get_nX() * geom2D.get_nY()) / 2] = 1.;

	Godzilla::Velocity2D vel2D(geom2D, scalar);
	Godzilla::Field2D forcing2D(geom2D, data);
	Godzilla::Field2D solution2D(geom2D);
	Godzilla::BoundaryCondition2D bc2D(geom2D, "PML", "PML", "PML", "PML");
	double omega = 1;

	Godzilla::Helmholtz2DReal::SparseDirectSolver2D solver(&vel2D, &forcing2D, &bc2D, omega, 0);
	solver.create_sparse_matrix_rhs();
	solver.solve();

	solver.extract_solution(solution2D);
	const Godzilla::xd *data_sol = solution2D.get_cdata().data();
	size_t nX = solution2D.get_geom2D().get_nX();
	size_t nY = solution2D.get_geom2D().get_nY();
	for (size_t i = 0; i < nY; ++i) {
		for (size_t j = 0; j < nX; ++j) {
			std::cout << data_sol[i * nX + j] << " ";
		}
		std::cout << std::endl;
	}

	int haltscreen;
	std::cin >> haltscreen;
	return 0;
}
