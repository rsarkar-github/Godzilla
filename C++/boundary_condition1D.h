#pragma once
#include "common.h"
#include <string>

namespace Godzilla {
	class BoundaryCondition1D {
		public:
			// Constructors
			BoundaryCondition1D() : _bc_face1(Godzilla::BC_DEFAULT), _bc_face2(Godzilla::BC_DEFAULT),
									_pmlcells_face1(Godzilla::PML_CELLS_DEFAULT), _pmlcells_face2(Godzilla::PML_CELLS_DEFAULT) {}

			BoundaryCondition1D(std::string bc_face1, std::string bc_face2 = Godzilla::BC_DEFAULT);
			BoundaryCondition1D(std::string bc_face1, size_t pmlcells_face1, std::string bc_face2 = Godzilla::BC_DEFAULT, size_t pmlcells_face2 = Godzilla::PML_CELLS_DEFAULT);

			BoundaryCondition1D(const Godzilla::BoundaryCondition1D &bc1D);
			BoundaryCondition1D(Godzilla::BoundaryCondition1D &&bc1D);

			// Public members
			BoundaryCondition1D& operator=(const Godzilla::BoundaryCondition1D &bc1D);
			BoundaryCondition1D& operator=(Godzilla::BoundaryCondition1D &&bc1D);
			std::string get_bc_face1() const { return _bc_face1; }
			std::string get_bc_face2() const { return _bc_face2; }
			size_t get_pmlcells_face1() const { return _pmlcells_face1; }
			size_t get_pmlcells_face2() const { return _pmlcells_face2; }

			void set_bc_face1(std::string &bc_face1) { _bc_face1 = bc_face1; }
			void set_bc_face2(std::string &bc_face2) { _bc_face2 = bc_face2; }
			void set_pmlcells_face1(size_t &pmlcells_face1) { _pmlcells_face1 = pmlcells_face1; }
			void set_pmlcells_face2(size_t &pmlcells_face2) { _pmlcells_face2 = pmlcells_face2; }

		private:
			// Private members
			// left and right faces
			std::string _bc_face1;   // id = 0
			std::string _bc_face2;   // id = 1

			size_t _pmlcells_face1;  // id = 3
			size_t _pmlcells_face2;  // id = 4
	};
}