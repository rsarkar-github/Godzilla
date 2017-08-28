#pragma once
#include <vector>
#include <complex>

namespace Godzilla {
	// Project typedefs
	typedef std::complex<double> xd;
	typedef std::vector<double> vecd;
	typedef std::vector<std::complex<double>> vecxd;

	// Numerical stability
	extern double VEL_MIN_TOL;
	extern double VEL_MAX_TOL;
	extern double GRID_MIN_SIZE;
	extern double GRID_MAX_SIZE;

	// PML parameters
	extern size_t PML_CELLS_DEFAULT;
	extern std::complex<double> PML_DAMPING;

	// Boundary conditions
	extern std::string BC_DEFAULT;

	// For lock manager
	extern int VELOCITY1D_DATA_ID;
	extern int VELOCITY2D_DATA_ID;
	extern int FIELD1D_DATA_ID;
	extern int FIELD2D_DATA_ID;
}