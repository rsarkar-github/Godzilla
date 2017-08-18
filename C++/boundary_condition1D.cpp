#include "boundary_condition1D.h"

namespace Godzilla {
	// Constructors
	BoundaryCondition1D::BoundaryCondition1D(std::string bc_face1, std::string bc_face2) {
		_bc_face1 = bc_face1;
		_bc_face2 = bc_face2;
		_pmlcells_face1 = Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face2 = Godzilla::PML_CELLS_DEFAULT;
	}

	BoundaryCondition1D::BoundaryCondition1D(std::string bc_face1, size_t pmlcells_face1, std::string bc_face2, size_t pmlcells_face2) {
		_bc_face1 = bc_face1;
		_bc_face2 = bc_face2;
		_pmlcells_face1 = ((bc_face1 == "PML") && (pmlcells_face1 > 0)) ? pmlcells_face1 : Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face2 = ((bc_face2 == "PML") && (pmlcells_face2 > 0)) ? pmlcells_face2 : Godzilla::PML_CELLS_DEFAULT;
	}

	BoundaryCondition1D::BoundaryCondition1D(const Godzilla::BoundaryCondition1D &bc1D) {
		_bc_face1 = bc1D.get_bc_face1();
		_bc_face2 = bc1D.get_bc_face2();
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
	}

	BoundaryCondition1D::BoundaryCondition1D(Godzilla::BoundaryCondition1D &&bc1D) {
		_bc_face1 = std::move(bc1D._bc_face1);
		_bc_face2 = std::move(bc1D._bc_face2);
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
	}

	// Public members
	Godzilla::BoundaryCondition1D& BoundaryCondition1D::operator=(const Godzilla::BoundaryCondition1D &bc1D) {
		_bc_face1 = bc1D.get_bc_face1();
		_bc_face2 = bc1D.get_bc_face2();
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
		return *this;
	}

	Godzilla::BoundaryCondition1D& BoundaryCondition1D::operator=(Godzilla::BoundaryCondition1D &&bc1D) {
		_bc_face1 = std::move(bc1D._bc_face1);
		_bc_face2 = std::move(bc1D._bc_face2);
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
		return *this;
	}
}