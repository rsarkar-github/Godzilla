#pragma once
#include "geometry1D.h"
#include <string>

namespace Godzilla {
	class BoundaryCondition1D {
		public:
			// Constructors
			BoundaryCondition1D();
			BoundaryCondition1D(const Godzilla::Geometry1D &geom1D);
			BoundaryCondition1D(const Godzilla::Geometry1D &geom1D, const std::string &bc_face1, const std::string &bc_face2 = Godzilla::BC_DEFAULT);
			BoundaryCondition1D(const Godzilla::Geometry1D &geom1D, const std::string &bc_face1, const size_t &pmlcells_face1,
								const std::string &bc_face2 = Godzilla::BC_DEFAULT, const size_t &pmlcells_face2 = Godzilla::PML_CELLS_DEFAULT);

			BoundaryCondition1D(const Godzilla::BoundaryCondition1D &bc1D);
			BoundaryCondition1D(Godzilla::BoundaryCondition1D &&bc1D);

			// Public methods
			BoundaryCondition1D& operator=(const Godzilla::BoundaryCondition1D &bc1D);
			BoundaryCondition1D& operator=(Godzilla::BoundaryCondition1D &&bc1D);
			std::string get_bc_face1() const { return _bc_face1; }
			std::string get_bc_face2() const { return _bc_face2; }
			size_t get_pmlcells_face1() const { return _pmlcells_face1; }
			size_t get_pmlcells_face2() const { return _pmlcells_face2; }
			Godzilla::Geometry1D get_geom1D() const { return _geom1D; }
			const Godzilla::xd get_data1() const { return _data1; }
			const Godzilla::xd get_data2() const { return _data2; }

			void set_bc_face1(const std::string &bc_face1);
			void set_bc_face2(const std::string &bc_face2);
			void set_pmlcells_face1(const size_t &pmlcells_face1);
			void set_pmlcells_face2(const size_t &pmlcells_face2);
			void set_data(const Godzilla::xd &data1, const Godzilla::xd &data2);

		private:
			// Private members
			// Boundry conditions for left and right faces
			std::string _bc_face1;   // id = 0
			std::string _bc_face2;   // id = 1

			// PML cell widths for left and right faces
			size_t _pmlcells_face1;  // id = 2
			size_t _pmlcells_face2;  // id = 3

			// 1D geometry
			Geometry1D _geom1D;      // id = 4

			// Data for left and right faces
			Godzilla::xd _data1;     // id = 5
			Godzilla::xd _data2;     // id = 6

			// Private methods
			bool is_valid_boundary_condition(const std::string &bcface) const;
			bool is_PML(const std::string &bc) const { return (bc == "PML") ? true : false; };
	};
}