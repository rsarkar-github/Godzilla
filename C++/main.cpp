#include <iostream>
#include <stdio.h>
#include "velocity2D.h"
#include "field2D.h"
#include "sparse_direct_solver2D.h"
#include "boundary_condition2D.h"
#include "umfpack.h"

int main() {

	size_t ncellsX = 3, ncellsY = 3;
	double startX = 0., startY = 0.;
	double endX = 10., endY = 10.;

	Godzilla::Geometry2D geom2D(startX, endX, ncellsX, startY, endY, ncellsY, "x", "y");

	Godzilla::xd scalar(5., 0.);
	Godzilla::vecd data(geom2D.get_nX() * geom2D.get_nY(), 0.);
	double cnt = 2;
	for (auto &i : data) {
		i = cnt;
		cnt += 1.0;
	}
	Godzilla::Velocity2D vel2D(geom2D, data);
	Godzilla::Field2D forcing2D(geom2D, 1.);
	Godzilla::BoundaryCondition2D bc2D(geom2D, "PML", "PML", "PML", "PML");
	double omega = 1;

	Godzilla::Helmholtz2DReal::SparseDirectSolver2D solver(&vel2D, &forcing2D, &bc2D, omega, 0);
	solver.create_sparse_matrix_rhs();
	solver.solve();

	int haltscreen;
	std::cin >> haltscreen;
	return 0;
}


//int main(void) {
//
//	SuiteSparse_long n = 5;
//	SuiteSparse_long Ap[] = { 0, 2, 5, 9, 10, 12 };
//	SuiteSparse_long Ai[] = { 0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4 };
//	double Ax[] = { 2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1. };
//	double Az[] = { 2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1. };
//	double bx[] = { 8., 45., -3., 3., 19. };
//	double bz[] = { 8., 45., -3., 3., 19. };
//	double Xx[5], Xz[5];
//
//	double *null = (double *)NULL;
//	int i;
//	void *Symbolic, *Numeric;
//	(void)umfpack_zl_symbolic(n, n, Ap, Ai, Ax, Az, &Symbolic, null, null);
//	(void)umfpack_zl_numeric(Ap, Ai, Ax, Az, Symbolic, &Numeric, null, null);
//	umfpack_zl_free_symbolic(&Symbolic);
//	(void)umfpack_zl_solve(UMFPACK_A, Ap, Ai, Ax, Az, Xx, Xz, bx, bz, Numeric, null, null);
//	umfpack_zl_free_numeric(&Numeric);
//	for (i = 0; i < n; i++) printf("Xx [%d] = %g, Xz [%d] = %g\n", i, Xx[i], Xz[i]);
//
//	int a;
//	std::cin >> a;
//
//	return 0;
//
//}