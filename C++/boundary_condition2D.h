#pragma once
#include "common.h"
#include <string>

namespace Godzilla {
	class BoundaryCondition2D {
		public:
			// Constructors
			BoundaryCondition2D() : _bc_face1(Godzilla::BC_DEFAULT), _bc_face2(Godzilla::BC_DEFAULT),
									_bc_face3(Godzilla::BC_DEFAULT), _bc_face4(Godzilla::BC_DEFAULT),
									_pmlcells_face1(Godzilla::PML_CELLS_DEFAULT), _pmlcells_face2(Godzilla::PML_CELLS_DEFAULT),
									_pmlcells_face3(Godzilla::PML_CELLS_DEFAULT), _pmlcells_face4(Godzilla::PML_CELLS_DEFAULT) {}
			
			BoundaryCondition2D(std::string bc_face1, std::string bc_face2 = Godzilla::BC_DEFAULT,
								std::string bc_face3 = Godzilla::BC_DEFAULT, std::string bc_face4 = Godzilla::BC_DEFAULT);

			BoundaryCondition2D(std::string bc_face1, size_t pmlcells_face1,
								std::string bc_face2 = Godzilla::BC_DEFAULT, size_t pmlcells_face2 = Godzilla::PML_CELLS_DEFAULT,
								std::string bc_face3 = Godzilla::BC_DEFAULT, size_t pmlcells_face3 = Godzilla::PML_CELLS_DEFAULT,
								std::string bc_face4 = Godzilla::BC_DEFAULT, size_t pmlcells_face4 = Godzilla::PML_CELLS_DEFAULT);

			BoundaryCondition2D(const Godzilla::BoundaryCondition2D &bc2D);
			BoundaryCondition2D(Godzilla::BoundaryCondition2D &&bc2D);

			// Public members
			BoundaryCondition2D& operator=(const Godzilla::BoundaryCondition2D &bc2D);
			BoundaryCondition2D& operator=(Godzilla::BoundaryCondition2D &&bc2D);
			std::string get_bc_face1() const { return _bc_face1; }
			std::string get_bc_face2() const { return _bc_face2; }
			std::string get_bc_face3() const { return _bc_face3; }
			std::string get_bc_face4() const { return _bc_face4; }
			size_t get_pmlcells_face1() const { return _pmlcells_face1; }
			size_t get_pmlcells_face2() const { return _pmlcells_face2; }
			size_t get_pmlcells_face3() const { return _pmlcells_face3; }
			size_t get_pmlcells_face4() const { return _pmlcells_face4; }

			void set_bc_face1(std::string &bc_face1) { _bc_face1 = bc_face1; }
			void set_bc_face2(std::string &bc_face2) { _bc_face2 = bc_face2; }
			void set_bc_face3(std::string &bc_face3) { _bc_face3 = bc_face3; }
			void set_bc_face4(std::string &bc_face4) { _bc_face4 = bc_face4; }
			void set_pmlcells_face1(size_t &pmlcells_face1) { _pmlcells_face1 = pmlcells_face1; }
			void set_pmlcells_face2(size_t &pmlcells_face2) { _pmlcells_face2 = pmlcells_face2; }
			void set_pmlcells_face3(size_t &pmlcells_face3) { _pmlcells_face3 = pmlcells_face3; }
			void set_pmlcells_face4(size_t &pmlcells_face4) { _pmlcells_face4 = pmlcells_face4; }

		private:
			// Private members
			// left, top, right and bottom faces
			std::string _bc_face1;   // id = 0
			std::string _bc_face2;   // id = 1
			std::string _bc_face3;   // id = 2
			std::string _bc_face4;   // id = 3

			size_t _pmlcells_face1;  // id = 4
			size_t _pmlcells_face2;  // id = 5
			size_t _pmlcells_face3;  // id = 6
			size_t _pmlcells_face4;  // id = 7
	};
}