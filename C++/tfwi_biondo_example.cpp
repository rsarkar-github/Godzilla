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
	int nsamples = 500;                      // number of time samples
	double peak_freq = 5;                    // peak frequency of Ricker wavelet in Hz
	double dt = 0.01;                        // time sampling interval in seconds
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
	size_t ncellsX = 600, ncellsY = 299;
	double startX = -3., startY = -0.8;
	double endX = 3., endY = 2.19;
	Godzilla::Geometry2D geom2D(startX, endX, ncellsX, startY, endY, ncellsY, "x", "y");
	size_t nelem = geom2D.get_nX() * geom2D.get_nY();
	size_t receiver_Y_pos = 230;       // Receivers at depth 1.5 km

	// Create constant velocity background (1 km/s)
	Godzilla::xd scalar(1, 0.);
	Godzilla::Velocity2D vel2D(geom2D, scalar);

	// Initialize forcing function, solution, extended Born solution and boundary conditions
	Godzilla::Field2D forcing2D(geom2D);
	Godzilla::Field2D solution2D(geom2D);
	Godzilla::Field2D solution_extBorn2D(geom2D);
	Godzilla::BoundaryCondition2D bc2D(geom2D, "PML", "PML", "PML", "PML");

	// Initialize SparseDirectSolver object
	// Initialize omega
	Godzilla::Helmholtz2DReal::SparseDirectSolver2D solver(&vel2D, &forcing2D, &bc2D, 0., 0);
	double omega = 0.;

	// Initialize LockManager to change forcing
	waveX::LockManager<Godzilla::Field2D, Godzilla::vecxd> lock(forcing2D, 1);

	// Initialize vector to store background Helmholtz data for set of receivers and also in time
	Godzilla::vecxd recdata_bkg_fft(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_bkg(geom2D.get_nX() * nsamples, 0.);

	// Get pointer to solution
	// Background Helmholtz data for receivers and in time
	const Godzilla::xd *ptr_solution2D = solution2D.get_cdata().data();
	Godzilla::xd *ptr_recdata_bkg_fft = recdata_bkg_fft.data();
	Godzilla::xd *ptr_recdata_bkg = recdata_bkg.data();

	// Define center of gaussian and standard deviation
	// Create gaussian smoothed field used to inject the forcing function
	// Write to file
	double x0 = 0., y0 = 0., sigma = 0.1;
	double x = 0., y = 0.;                 // temporary variables to hold x, y coordinates
	Godzilla::vecd smooth_data(nelem, 0.);
	double *ptr_smooth_data = smooth_data.data();
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

	// Allocate pointers and objects for fft and writing files
	fftw_complex *in = nullptr, *out= nullptr;
	wavemod2d::EasyIO io;
	Godzilla::vecd real(geom2D.get_nX() * nsamples, 0.);
	double *ptr_real = real.data();

	// Solve the Helmholtz equation for different omega values (exclude omega = 0)
	size_t nsample_start = 1;
	size_t nsample_end = 75;
	//for (size_t i = nsample_start; i <= nsample_end; ++i) {
	//	// Print a message
	//	std::cout << "Modeling for i = " << i << std::endl;

	//	// Get new omega and update omega in solver
	//	omega = 2 * pi * ptr_freq_list[i];
	//	solver.change_omega(omega);

	//	// Change forcing function and update forcing function in solver
	//	lock.activate_lock(forcing2D);
	//	Godzilla::xd *ptr_forcing_data = lock._ptr->data();
	//	for (size_t j = 0; j < nelem; ++j) {
	//		ptr_forcing_data[j] = -ptr_wavelet_fft[i] * ptr_smooth_data[j];
	//	}
	//	lock.deactivate_lock(forcing2D);
	//	solver.change_forcing_data(&forcing2D);

	//	// Recreate sparse matrix
	//	solver.create_sparse_matrix_rhs();

	//	// Solve the system
	//	solver.solve();

	//	// Extract solution
	//	solver.extract_solution(solution2D);

	//	// Extract solution at receivers
	//	for (size_t j = 0; j < geom2D.get_nX(); ++j) {
	//		ptr_recdata_bkg_fft[j * nsamples + i] = ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j];
	//		ptr_recdata_bkg_fft[(j + 1) * nsamples - i] = std::conj(ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j]);
	//	}
	//}

	//// Fourier transform the background Helmholtz solution at receivers
	//in = reinterpret_cast<fftw_complex*>(ptr_recdata_bkg_fft);
	//out = reinterpret_cast<fftw_complex*>(ptr_recdata_bkg);
	//p = fftw_plan_dft_1d(nsamples, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	//for (size_t i = 0; i < geom2D.get_nX(); ++i) {
	//	fftw_execute_dft(p, in, out);
	//	in += nsamples;
	//	out += nsamples;
	//}
	//fftw_destroy_plan(p);

	//// Write to file background Hemlholtz solution
	//for (size_t i = 0; i < geom2D.get_nX() * nsamples; ++i) {
	//	ptr_real[i] = ptr_recdata_bkg[i].real();
	//}
	//io.write_binary("D:/Research/Freq-Domain/Godzilla/test/tfwi_biondo_bkg.bin", real);
	///////////////////////////////////////////////////////////////////////////////////

	// Read perturbed velocity and set as new velocity
	std::vector<float> vel_data = io.read_binary("D:/Research/Freq-Domain/Godzilla/test/velplusdvel.bin");
	Godzilla::vecd vel_pert_data(nelem, 0.);
	//for (size_t i = 0; i < nelem; ++i) {
	//	vel_pert_data[i] = vel_data[i];
	//}
	//vel2D.set_data(vel_pert_data);
	//solver.change_velocity_data(&vel2D);

	//// Initialize vector to store perturbed Helmholtz data for set of receivers and also in time
	//Godzilla::vecxd recdata_pert_fft(geom2D.get_nX() * nsamples, 0.);
	//Godzilla::vecxd recdata_pert(geom2D.get_nX() * nsamples, 0.);

	//// Perturbed Helmholtz data for receivers and in time
	//Godzilla::xd *ptr_recdata_pert_fft = recdata_pert_fft.data();
	//Godzilla::xd *ptr_recdata_pert = recdata_pert.data();

	//// Solve the Helmholtz equation for different omega values (exclude omega = 0)
	//for (size_t i = nsample_start; i <= nsample_end; ++i) {
	//	// Print a message
	//	std::cout << "Modeling for i = " << i << std::endl;

	//	// Get new omega and update omega in solver
	//	omega = 2 * pi * ptr_freq_list[i];
	//	solver.change_omega(omega);

	//	// Change forcing function and update forcing function in solver
	//	lock.activate_lock(forcing2D);
	//	Godzilla::xd *ptr_forcing_data = lock._ptr->data();
	//	for (size_t j = 0; j < geom2D.get_nX() * geom2D.get_nY(); ++j) {
	//		ptr_forcing_data[j] = -ptr_wavelet_fft[i] * ptr_smooth_data[j];
	//	}
	//	lock.deactivate_lock(forcing2D);
	//	solver.change_forcing_data(&forcing2D);

	//	// Recreate sparse matrix
	//	solver.create_sparse_matrix_rhs();

	//	// Solve the system
	//	solver.solve();

	//	// Extract solution
	//	solver.extract_solution(solution2D);

	//	// Extract solution at receivers
	//	for (size_t j = 0; j < geom2D.get_nX(); ++j) {
	//		ptr_recdata_pert_fft[j * nsamples + i] = ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j];
	//		ptr_recdata_pert_fft[(j + 1) * nsamples - i] = std::conj(ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j]);
	//	}
	//}

	//// Fourier transform the perturbed Helmholtz solution at receivers
	//in = reinterpret_cast<fftw_complex*>(ptr_recdata_pert_fft);
	//out = reinterpret_cast<fftw_complex*>(ptr_recdata_pert);
	//p = fftw_plan_dft_1d(nsamples, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	//for (size_t i = 0; i < geom2D.get_nX(); ++i) {
	//	fftw_execute_dft(p, in, out);
	//	in += nsamples;
	//	out += nsamples;
	//}
	//fftw_destroy_plan(p);

	//// Write to file perturbed Hemlholtz solution
	//for (size_t i = 0; i < geom2D.get_nX() * nsamples; ++i) {
	//	ptr_real[i] = ptr_recdata_pert[i].real();
	//}
	//io.write_binary("D:/Research/Freq-Domain/Godzilla/test/tfwi_biondo_pert.bin", real);
	///////////////////////////////////////////////////////////////////////////////////

	// Read in extended model
	size_t nlag = 65;
	double t0 = -0.32;
	Godzilla::xd phase(0., -2 * pi * t0);
	vel_data = io.read_binary("D:/Research/Freq-Domain/Godzilla/test/velextended.bin");
	vel_pert_data.clear();
	vel_pert_data.assign(vel_data.size(), 0.);
	if (vel_data.size() != nelem * nlag) {
		std::cerr << "Extended model size does not match program paraneters." << std::endl;
		assert(1 == 2);
	}
	for (size_t i = 0; i < nelem * nlag; ++i) {
		vel_pert_data[i] = vel_data[i];
	}

	// Create array to store frequency slices of extended model
	// Calculate FFT of extended model
	Godzilla::vecxd vel_extended_fft(nelem * (nsample_end - nsample_start + 1), 0.);
	Godzilla::vecxd in_ext(nsamples, 0.);
	Godzilla::vecxd out_ext(nsamples, 0.);

	Godzilla::xd *ptr_vel_extended_fft = vel_extended_fft.data();
	in = reinterpret_cast<fftw_complex*>(in_ext.data());
	out = reinterpret_cast<fftw_complex*>(out_ext.data());

	p = fftw_plan_dft_1d(nsamples, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	for (size_t i = 0; i < nelem; ++i) {
		for (size_t j = 0; j < nlag; ++j) {
			in_ext[j] = vel_pert_data[i * nlag + j];
		}
		fftw_execute_dft(p, in, out);
		for (size_t j = 0; j < nsamples; ++j) {
			out_ext[j] *= std::exp(phase * ptr_freq_list[j]);
		}
		for (size_t j = nsample_start; j <= nsample_end; ++j) {
			ptr_vel_extended_fft[(j - nsample_start) * nelem + i] = out_ext[j];
		}
	}
	fftw_destroy_plan(p);

	// Set velocity to background velocity
	vel2D = Godzilla::Velocity2D(geom2D, scalar);
	solver.change_velocity_data(&vel2D);

	// Initialize vector to store extended Born Helmholtz data for set of receivers and also in time
	Godzilla::vecxd recdata_extBorn_fft(geom2D.get_nX() * nsamples, 0.);
	Godzilla::vecxd recdata_extBorn(geom2D.get_nX() * nsamples, 0.);

	// Pointer for extended Born Helmholtz data for receivers and in time
	Godzilla::xd *ptr_recdata_extBorn_fft = recdata_extBorn_fft.data();
	Godzilla::xd *ptr_recdata_extBorn = recdata_extBorn.data();

	// Solve Helmholtz for different values of omega (except omega = 0)
	for (size_t i = nsample_start; i <= nsample_end; ++i) {
		// Print a message
		std::cout << "Modeling for i = " << i << std::endl;

		// Get new omega and update omega in solver
		omega = 2 * pi * ptr_freq_list[i];
		solver.change_omega(omega);

		// Change forcing function and update forcing function in solver
		lock.activate_lock(forcing2D);
		Godzilla::xd *ptr_forcing_data = lock._ptr->data();
		for (size_t j = 0; j < nelem; ++j) {
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

		// Change forcing for extended Born Helmholtz
		lock.activate_lock(forcing2D);
		ptr_forcing_data = lock._ptr->data();
		for (size_t j = 0; j < nelem; ++j) {
			ptr_forcing_data[j] = (2.0 * omega * omega / std::pow(scalar, 3.0)) * ptr_solution2D[j] * ptr_vel_extended_fft[j];
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
			ptr_recdata_extBorn_fft[j * nsamples + i] = ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j];
			ptr_recdata_extBorn_fft[(j + 1) * nsamples - i] = std::conj(ptr_solution2D[receiver_Y_pos * geom2D.get_nX() + j]);
		}

		// Advance pointer to vel_extended_fft
		ptr_vel_extended_fft += nelem;
	}

	// Fourier transform the extended Born Helmholtz solution at receivers
	in = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn_fft);
	out = reinterpret_cast<fftw_complex*>(ptr_recdata_extBorn);
	p = fftw_plan_dft_1d(nsamples, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	for (size_t i = 0; i < geom2D.get_nX(); ++i) {
		fftw_execute_dft(p, in, out);
		in += nsamples;
		out += nsamples;
	}
	fftw_destroy_plan(p);

	// Write to file background Hemlholtz solution
	for (size_t i = 0; i < geom2D.get_nX() * nsamples; ++i) {
		ptr_real[i] = ptr_recdata_extBorn[i].real();
	}
	io.write_binary("D:/Research/Freq-Domain/Godzilla/test/tfwi_biondo_extBorn.bin", real);

	return 0;
}