#include "common.h"
#include <string>

namespace Godzilla {
	// Numerical stability
	double VEL_MIN_TOL = 1e-6;
	double VEL_MAX_TOL = 1e6;
	double GRID_MIN_SIZE = 1e-6;
	double GRID_MAX_SIZE = 1e6;

	// PML parameters
	size_t PML_CELLS_DEFAULT = 10;

	// Boundary conditions
	std::string BC_DEFAULT = "DBC";

	// For lock manager
	int VELOCITY1D_DATA_ID = 1;
	int VELOCITY2D_DATA_ID = 1;
	int FIELD1D_DATA_ID = 1;
	int FIELD2D_DATA_ID = 1;
}