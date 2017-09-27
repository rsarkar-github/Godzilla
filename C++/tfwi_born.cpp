#include <iostream>
#include <stdio.h>
#include <cmath>
#include "easy_io.h"
#include "lock_manager.h"
#include "velocity2D.h"
#include "field2D.h"
#include "sparse_direct_solver2D.h"
#include "boundary_condition2D.h"
#include "umfpack.h"
#include "fftw3.h"

int main() {

	double pi = 3.14159265358979;

	// Define Ricker wavelet parameters
	// Create Ricker wavelet
	// Calculate FFT of Ricker wavelet
	// Define vector of omegas
	int nsamples = 250;                      // number of time samples
	double peak_freq = 10;                    // peak frequency of Ricker wavelet in Hz
	double dt = 0.008;                       // time sampling interval in seconds
	double delay_time = 0.2;                 // delay time of Ricker wavelet in seconds
	double dfreq = 1 / (dt * nsamples);      // frequency sampling interval

	Godzilla::vecxd wavelet(nsamples, 0.);
	Godzilla::vecxd wavelet_fft(nsamples, 0.);
	Godzilla::vecd freq_list(nsamples, 0.);
	Godzilla::xd *ptr_wavelet = wavelet.data();
	Godzilla::xd *ptr_wavelet_fft = wavelet_fft.data();
	double *ptr_freq_list = freq_list.data();
	double temp = 0.;
	for (size_t i = 0; i < nsamples; ++i) {
		temp = std::pow(pi * peak_freq * (i * dt - delay_time), 2.0);
		ptr_wavelet[i] = (1 - 2 * temp) * std::exp(-temp);
	}
	for (size_t i = 0; i <= nsamples / 2; ++i) {
		ptr_freq_list[i] = i * dfreq;
	}
	for (size_t i = 1; i < nsamples / 2; ++i) {
		ptr_freq_list[nsamples - i] = -(i * dfreq);
	}
	fftw_plan p;
	p = fftw_plan_dft_1d(nsamples, reinterpret_cast<fftw_complex*>(ptr_wavelet), reinterpret_cast<fftw_complex*>(ptr_wavelet_fft), FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);

	// Define simulation geometry
	// Create geometry object
	// Define source injection position
	// Define position of receivers along Y
	size_t ncellsX = 500, ncellsY = 500;
	double startX = 0., startY = 0.;
	double endX = 10., endY = 10.;
	Godzilla::Geometry2D geom2D(startX, endX, ncellsX, startY, endY, ncellsY, "x", "y");
	size_t source_pos = geom2D.get_nX() * geom2D.get_nY() / 2;
	size_t receiver_Y_pos = 3 * geom2D.get_nY() / 4;

	// Create constant velocity background (5 km/s)
	Godzilla::xd scalar(5, 0.);
	Godzilla::Velocity2D vel2D(geom2D, scalar);

	// Initialize forcing function, solution, extended Born solution and boundary conditions
	Godzilla::Field2D forcing2D(geom2D);
	Godzilla::Field2D solution2D(geom2D);
	Godzilla::Field2D solution_extBorn2D(geom2D);
	Godzilla::BoundaryCondition2D bc2D(geom2D, "PML", "PML", "PML", "PML");

	// Initialize SparseDirectSolver object
	// Initailize omega
	Godzilla::Helmholtz2DReal::SparseDirectSolver2D solver(&vel2D, &forcing2D, &bc2D, 0., 0);
	double omega = 0.;

	// Initialize LockManager to change forcing
	waveX::LockManager<Godzilla::Field2D, Godzilla::vecxd> lock(forcing2D, 1);

	// Initialize vector to store background Helmholtz data for set of receivers and also in time
	// Initialize vector to store extended Born Helmholtz data for set of receivers and also in time (case1)
	// Initialize vector to store extended Born Helmholtz data for set of receivers and also in time (case2)
	// Initialize vector to store extended Born Helmholtz data for set of receivers and also in time (case3)
	Godzilla::vecxd recdata_bkg_fft(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_bkg(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_extBorn1_fft(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_extBorn1(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_extBorn2_fft(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_extBorn2(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_extBorn3_fft(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_extBorn3(geom2D.get_nX() * nsamples, 0.);

	// Get pointer to solution and solution for extended Born
	// Background Helmholtz data for receivers and in time
	// Extended Born Helmholtz data for receivers and in time (Case1)
	// Extended Born Helmholtz data for receivers and in time (Case2)
	// Extended Born Helmholtz data for receivers and in time (Case3)
	const Godzilla::xd *ptr_solution2D = solution2D.get_cdata().data();
	const Godzilla::xd *ptr_solution_extBorn2D = solution_extBorn2D.get_cdata().data();
	Godzilla::xd *ptr_recdata_bkg_fft = recdata_bkg_fft.data();
	Godzilla::xd *ptr_recdata_bkg = recdata_bkg.data();
	Godzilla::xd *ptr_recdata_extBorn1_fft = recdata_extBorn1_fft.data();
	Godzilla::xd *ptr_recdata_extBorn1 = recdata_extBorn1.data();
	Godzilla::xd *ptr_recdata_extBorn2_fft = recdata_extBorn2_fft.data();
	Godzilla::xd *ptr_recdata_extBorn2 = recdata_extBorn2.data();
	Godzilla::xd *ptr_recdata_extBorn3_fft = recdata_extBorn3_fft.data();
	Godzilla::xd *ptr_recdata_extBorn3 = recdata_extBorn3.data();

	// Define center of gaussian and standard deviation
	// Define center of gaussian and standard deviation for perturbation
	// Create gaussian smoothed field used to inject the forcing function
	// Create gaussian smoothed field used for perturbation
	// Write to file
	double x0 = 5., y0 = 5., sigma = 0.1;
	double x0_pert = 5., y0_pert = 6.5, sigma_pert = 0.1;
	double x = 0., y = 0.;                 // temporary variables to hold x, y coordinates
	Godzilla::vecd smooth_data(geom2D.get_nX() * geom2D.get_nY(), 0.);
	Godzilla::vecd smooth_pert_data(geom2D.get_nX() * geom2D.get_nY(), 0.);
	double *ptr_smooth_data = smooth_data.data();
	double *ptr_smooth_pert_data = smooth_pert_data.data();
	double hX = geom2D.get_hX(), hY = geom2D.get_hY();
	double f1 = 0.5 / std::pow(sigma, 2.), f2 = std::sqrt(f1 / pi);
	y = startY;
	for (size_t i = 0; i < geom2D.get_nY(); ++i) {
		x = startX;
		for (size_t j = 0; j < geom2D.get_nX(); ++j) {
			ptr_smooth_data[j] = f2 * std::exp(-f1 * ((x - x0) * (x - x0) + (y - y0) * (y - y0)));
			x += hX;
		}
		y += hY;
		ptr_smooth_data += geom2D.get_nX();
	}
	ptr_smooth_data = smooth_data.data();  // Reset pointer
	wavemod2d::EasyIO io;
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/gauss_smooth.bin", smooth_data);

	f1 = 0.5 / std::pow(sigma_pert, 2.), f2 = std::sqrt(f1 / pi);
	y = startY;
	for (size_t i = 0; i < geom2D.get_nY(); ++i) {
		x = startX;
		for (size_t j = 0; j < geom2D.get_nX(); ++j) {
			ptr_smooth_pert_data[j] = f2 * std::exp(-f1 * ((x - x0_pert) * (x - x0_pert) + (y - y0_pert) * (y - y0_pert)));
			x += hX;
		}
		y += hY;
		ptr_smooth_pert_data += geom2D.get_nX();
	}
	ptr_smooth_pert_data = smooth_pert_data.data();  // Reset pointer
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/gauss_pert_smooth.bin", smooth_pert_data);

	// Define time delays
	// Phase factors for extended Born Helmholtz (Case1, Case2, Case3)
	double delay1 = 0., delay2 = 0.5, delay3 = -0.5;
	Godzilla::xd phase1(0, 0);
	Godzilla::xd phase2(0, 0);
	Godzilla::xd phase3(0, 0);

	// Solve the Helmholtz equation for different omega values (exclude omega = 0)
	size_t nsample_start = 1;
	size_t nsample_end = 60;
	for (size_t i = nsample_start; i <= nsample_end; ++i) {
		// Print a message
		std::cout << "Modeling for i = " << i << std::endl;

		// Get new omega and update omega in solver
		omega = 2 * pi * ptr_freq_list[i];
		solver.change_omega(omega);

		// Change forcing function and update forcing function in solver
		lock.activate_lock(forcing2D);
		Godzilla::xd *ptr_forcing_data = lock._ptr->data();
		for (size_t j = 0; j < geom2D.get_nX() * geom2D.get_nY(); ++j) {
			ptr_forcing_data[j] = -ptr_wavelet_fft[i] * ptr_smooth_data[j];
		}
		lock.deactivate_lock(forcing2D);
		solver.change_forcing_data(&forcing2D);

		// Recreate sparse matrix
		solver.create_sparse_matrix_rhs();

		// Solve the system
		solver.solve();

		// Extract solution
		solver.extract_solution(solution2D);

		// Extract solution at receivers
		for (size_t j = 0; j < geom2D.get_nX(); ++j) {
			ptr_recdata_bkg_fft[j * nsamples + i] = ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j];
			ptr_recdata_bkg_fft[(j + 1) * nsamples - i] = std::conj(ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j]);
		}

		// Change forcing for Case1 extended Born Helmholtz
		phase1 = std::exp(Godzilla::xd(0., delay1 * omega));
		lock.activate_lock(forcing2D);
		ptr_forcing_data = lock._ptr->data();
		for (size_t j = 0; j < geom2D.get_nX() * geom2D.get_nY(); ++j) {
			ptr_forcing_data[j] = (2.0 * omega * omega / std::pow(scalar, 3.0)) * ptr_solution2D[j] * phase1 * ptr_smooth_pert_data[j];
		}
		lock.deactivate_lock(forcing2D);
		solver.change_forcing_data(&forcing2D);

		// Recreate sparse matrix
		solver.create_sparse_matrix_rhs();

		// Solve the system
		solver.solve();

		// Extract solution
		solver.extract_solution(solution_extBorn2D);

		// Extract solution at receivers
		for (size_t j = 0; j < geom2D.get_nX(); ++j) {
			ptr_recdata_extBorn1_fft[j * nsamples + i] = ptr_solution_extBorn2D[receiver_Y_pos * geom2D.get_nX() + j];
			ptr_recdata_extBorn1_fft[(j + 1) * nsamples - i] = std::conj(ptr_solution_extBorn2D[receiver_Y_pos * geom2D.get_nX() + j]);
		}

		// Change forcing for Case2 extended Born Helmholtz
		phase2 = std::exp(Godzilla::xd(0., delay2 * omega));
		lock.activate_lock(forcing2D);
		ptr_forcing_data = lock._ptr->data();
		for (size_t j = 0; j < geom2D.get_nX() * geom2D.get_nY(); ++j) {
			ptr_forcing_data[j] = (2.0 * omega * omega / std::pow(scalar, 3.0)) * ptr_solution2D[j] * phase2 * ptr_smooth_pert_data[j];
		}
		lock.deactivate_lock(forcing2D);
		solver.change_forcing_data(&forcing2D);

		// Recreate sparse matrix
		solver.create_sparse_matrix_rhs();

		// Solve the system
		solver.solve();

		// Extract solution
		solver.extract_solution(solution_extBorn2D);

		// Extract solution at receivers
		for (size_t j = 0; j < geom2D.get_nX(); ++j) {
			ptr_recdata_extBorn2_fft[j * nsamples + i] = ptr_solution_extBorn2D[receiver_Y_pos * geom2D.get_nX() + j];
			ptr_recdata_extBorn2_fft[(j + 1) * nsamples - i] = std::conj(ptr_solution_extBorn2D[receiver_Y_pos * geom2D.get_nX() + j]);
		}

		// Change forcing for Case3 extended Born Helmholtz
		phase3 = std::exp(Godzilla::xd(0., delay3 * omega));
		lock.activate_lock(forcing2D);
		ptr_forcing_data = lock._ptr->data();
		for (size_t j = 0; j < geom2D.get_nX() * geom2D.get_nY(); ++j) {
			ptr_forcing_data[j] = (2.0 * omega * omega / std::pow(scalar, 3.0)) * ptr_solution2D[j] * phase3 * ptr_smooth_pert_data[j];
		}
		lock.deactivate_lock(forcing2D);
		solver.change_forcing_data(&forcing2D);

		// Recreate sparse matrix
		solver.create_sparse_matrix_rhs();

		// Solve the system
		solver.solve();

		// Extract solution
		solver.extract_solution(solution_extBorn2D);

		// Extract solution at receivers
		for (size_t j = 0; j < geom2D.get_nX(); ++j) {
			ptr_recdata_extBorn3_fft[j * nsamples + i] = ptr_solution_extBorn2D[receiver_Y_pos * geom2D.get_nX() + j];
			ptr_recdata_extBorn3_fft[(j + 1) * nsamples - i] = std::conj(ptr_solution_extBorn2D[receiver_Y_pos * geom2D.get_nX() + j]);
		}
	}

	// Fourier transform the background Helmholtz solution at receivers
	fftw_complex *in = reinterpret_cast<fftw_complex*>(ptr_recdata_bkg_fft);
	fftw_complex *out = reinterpret_cast<fftw_complex*>(ptr_recdata_bkg);
	p = fftw_plan_dft_1d(nsamples, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	for (size_t i = 0; i < geom2D.get_nX(); ++i) {
		fftw_execute_dft(p, in, out);
		in += nsamples;
		out += nsamples;
	}

	// Fourier transform the extended Born Helmholtz solution at receivers (Case1)
	in = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn1_fft);
	out = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn1);
	for (size_t i = 0; i < geom2D.get_nX(); ++i) {
		fftw_execute_dft(p, in, out);
		in += nsamples;
		out += nsamples;
	}
	// Fourier transform the extended Born Helmholtz solution at receivers (Case2)
	in = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn2_fft);
	out = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn2);
	for (size_t i = 0; i < geom2D.get_nX(); ++i) {
		fftw_execute_dft(p, in, out);
		in += nsamples;
		out += nsamples;
	}
	// Fourier transform the extended Born Helmholtz solution at receivers (Case3)
	in = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn3_fft);
	out = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn3);
	for (size_t i = 0; i < geom2D.get_nX(); ++i) {
		fftw_execute_dft(p, in, out);
		in += nsamples;
		out += nsamples;
	}
	fftw_destroy_plan(p);

	// Write to file all solutions
	Godzilla::vecd real(geom2D.get_nX() * nsamples, 0.);
	double *ptr_real = real.data();
	for (size_t i = 0; i < geom2D.get_nX() * nsamples; ++i) {
		ptr_real[i] = ptr_recdata_bkg[i].real();
	}
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/real_reconstruction.bin", real);

	for (size_t i = 0; i < geom2D.get_nX() * nsamples; ++i) {
		ptr_real[i] = ptr_recdata_extBorn1[i].real();
	}
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/real_extBorn1_reconstruction.bin", real);

	for (size_t i = 0; i < geom2D.get_nX() * nsamples; ++i) {
		ptr_real[i] = ptr_recdata_extBorn2[i].real();
	}
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/real_extBorn2_reconstruction.bin", real);

	for (size_t i = 0; i < geom2D.get_nX() * nsamples; ++i) {
		ptr_real[i] = ptr_recdata_extBorn3[i].real();
	}
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/real_extBorn3_reconstruction.bin", real);

	return 0;
}