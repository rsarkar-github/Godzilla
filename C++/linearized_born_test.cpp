#include <iostream>
#include <stdio.h>
#include <cmath>
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

	Godzilla::xd vel_scalar(1., 0.);

	Godzilla::vecd forcing_data(geom2D.get_nX() * geom2D.get_nY(), 0.);
	forcing_data[(geom2D.get_nX() * geom2D.get_nY()) / 2] = 1.;

	Godzilla::vecxd velpert_data(geom2D.get_nX() * geom2D.get_nY(), vel_scalar);
	velpert_data[geom2D.get_nX() * (geom2D.get_nY() / 4) + geom2D.get_nX() / 2] *= 1.1;

	Godzilla::Velocity2D vel2D_bkg(geom2D, vel_scalar);
	Godzilla::Velocity2D vel2D_pert(geom2D, velpert_data);
	Godzilla::Field2D forcing2D(geom2D, forcing_data);

	Godzilla::Field2D solution2D_bkg(geom2D);
	Godzilla::Field2D solution2D_pert(geom2D);

	Godzilla::BoundaryCondition2D bc2D(geom2D, "PML", "PML", "PML", "PML");
	double omega = 20;

	Godzilla::Helmholtz2DReal::SparseDirectSolver2D solver(&vel2D_bkg, &forcing2D, &bc2D, omega, 0);
	solver.create_sparse_matrix_rhs();
	solver.solve();
	solver.extract_solution(solution2D_bkg);

	solver.change_velocity_data(&vel2D_pert);
	solver.create_sparse_matrix_rhs();
	solver.solve();
	solver.extract_solution(solution2D_pert);

	Godzilla::vecd real, imag;
	size_t nelem = solution2D_bkg.get_nelem();
	real.assign(nelem, 0.);
	imag.assign(nelem, 0.);

	const Godzilla::xd *ptr_sol_bkg = solution2D_bkg.get_cdata().data();
	const Godzilla::xd *ptr_sol_pert = solution2D_pert.get_cdata().data();
	double *ptr_real = real.data();
	double *ptr_imag = imag.data();
	for (size_t i = 0; i < nelem; ++i) {
		ptr_real[i] = ptr_sol_pert[i].real() - ptr_sol_bkg[i].real();
		ptr_imag[i] = ptr_sol_pert[i].imag() - ptr_sol_bkg[i].imag();
	}

	wavemod2d::EasyIO io;
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/real_diff.bin", real);
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/imag_diff.bin", imag);

	Godzilla::vecxd born_forcing_data(geom2D.get_nX() * geom2D.get_nY(), 0.);
	born_forcing_data[geom2D.get_nX() * (geom2D.get_nY() / 4) + geom2D.get_nX() / 2] =
		2 * omega * omega * (1. / 1.) * solution2D_bkg.get_cdata()[geom2D.get_nX() * (geom2D.get_nY() / 4) + geom2D.get_nX() / 2];

	Godzilla::Field2D forcing2D_born(geom2D, born_forcing_data);
	Godzilla::Field2D solution2D_born(geom2D);

	solver.change_velocity_data(&vel2D_bkg);
	solver.change_forcing_data(&forcing2D_born);
	solver.create_sparse_matrix_rhs();
	solver.solve();
	solver.extract_solution(solution2D_born);

	const Godzilla::xd *ptr_sol_born = solution2D_born.get_cdata().data();
	ptr_real = real.data();
	ptr_imag = imag.data();
	for (size_t i = 0; i < nelem; ++i) {
		ptr_real[i] = ptr_sol_born[i].real();
		ptr_imag[i] = ptr_sol_born[i].imag();
	}
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/real_born.bin", real);
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/imag_born.bin", imag);

	// Linearization test
	size_t N = 10;
	std::vector<double> norm(10, 0);
	for (size_t i1 = 1; i1 <= N; ++i1) {
		velpert_data[geom2D.get_nX() * (geom2D.get_nY() / 4) + geom2D.get_nX() / 2] = 1.0 + double(i1) / 100;
		vel2D_pert.set_data(velpert_data);

		solver.change_velocity_data(&vel2D_pert);
		solver.change_forcing_data(&forcing2D);
		solver.create_sparse_matrix_rhs();
		solver.solve();
		solver.extract_solution(solution2D_pert);

		born_forcing_data[geom2D.get_nX() * (geom2D.get_nY() / 4) + geom2D.get_nX() / 2] =
			2 * omega * omega * ((double(i1) / 100) / 1.) * solution2D_bkg.get_cdata()[geom2D.get_nX() * (geom2D.get_nY() / 4) + geom2D.get_nX() / 2];
		forcing2D_born.set_data(born_forcing_data);

		solver.change_velocity_data(&vel2D_bkg);
		solver.change_forcing_data(&forcing2D_born);
		solver.create_sparse_matrix_rhs();
		solver.solve();
		solver.extract_solution(solution2D_born);

		ptr_sol_pert = solution2D_pert.get_cdata().data();
		ptr_sol_born = solution2D_born.get_cdata().data();

		for (size_t i = 0; i < nelem; ++i) {
			norm[i1-1] += std::pow(ptr_sol_born[i].real() - ptr_sol_pert[i].real() + ptr_sol_bkg[i].real(), 2) + 
						  std::pow(ptr_sol_born[i].imag() - ptr_sol_pert[i].imag() + ptr_sol_bkg[i].imag(), 2);
		}
		norm[i1-1] = std::sqrt(norm[i1-1]);
		std::cout << "% change = " << i1 << ", norm = " << norm[i1 - 1] << std::endl;
	}

	int haltScreen;
	std::cin >> haltScreen;
	return 0;
}
