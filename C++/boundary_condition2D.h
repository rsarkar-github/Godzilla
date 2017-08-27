#pragma once
#include "geometry2D.h"
#include <string>

namespace Godzilla {
	class BoundaryCondition2D {
		public:
			// Constructors
			BoundaryCondition2D();

			BoundaryCondition2D(const Godzilla::Geometry2D &geom2D);
			
			BoundaryCondition2D(const Godzilla::Geometry2D &geom2D, const std::string &bc_face1, const std::string &bc_face2 = Godzilla::BC_DEFAULT,
								const std::string &bc_face3 = Godzilla::BC_DEFAULT, const std::string &bc_face4 = Godzilla::BC_DEFAULT);

			BoundaryCondition2D(const Godzilla::Geometry2D &geom2D, const std::string &bc_face1, const size_t &pmlcells_face1,
								const std::string &bc_face2 = Godzilla::BC_DEFAULT, const size_t &pmlcells_face2 = Godzilla::PML_CELLS_DEFAULT,
								const std::string &bc_face3 = Godzilla::BC_DEFAULT, const size_t &pmlcells_face3 = Godzilla::PML_CELLS_DEFAULT,
								const std::string &bc_face4 = Godzilla::BC_DEFAULT, const size_t &pmlcells_face4 = Godzilla::PML_CELLS_DEFAULT);

			BoundaryCondition2D(const Godzilla::BoundaryCondition2D &bc2D);
			BoundaryCondition2D(Godzilla::BoundaryCondition2D &&bc2D);

			// Public methods
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
			Godzilla::Geometry2D get_geom2D() const { return _geom2D; }
			const Godzilla::vecxd& get_data1() const { return _data1; }
			const Godzilla::vecxd& get_data2() const { return _data2; }
			const Godzilla::vecxd& get_data3() const { return _data3; }
			const Godzilla::vecxd& get_data4() const { return _data4; }

			void set_bc_face1(const std::string &bc_face1);
			void set_bc_face2(const std::string &bc_face2);
			void set_bc_face3(const std::string &bc_face3);
			void set_bc_face4(const std::string &bc_face4);
			void set_pmlcells_face1(const size_t &pmlcells_face1);
			void set_pmlcells_face2(const size_t &pmlcells_face2);
			void set_pmlcells_face3(const size_t &pmlcells_face3);
			void set_pmlcells_face4(const size_t &pmlcells_face4);
			void set_data(const Godzilla::vecxd &data1, const Godzilla::vecxd &data2,
						  const Godzilla::vecxd &data3, const Godzilla::vecxd &data4);

		private:
			// Private members
			// Boundary condition type for left, top, right and bottom faces
			std::string _bc_face1;   // id = 0
			std::string _bc_face2;   // id = 1
			std::string _bc_face3;   // id = 2
			std::string _bc_face4;   // id = 3

			// PML cell widths for left, top, right and bottom faces
			size_t _pmlcells_face1;  // id = 4
			size_t _pmlcells_face2;  // id = 5
			size_t _pmlcells_face3;  // id = 6
			size_t _pmlcells_face4;  // id = 7

			// 2D geometry
			Geometry2D _geom2D;      // id = 8

			// Data for left, top, right and bottom faces
			Godzilla::vecxd _data1;  // id = 9
			Godzilla::vecxd _data2;  // id = 10
			Godzilla::vecxd _data3;  // id = 11
			Godzilla::vecxd _data4;  // id = 12

			// Private methods
			bool is_valid_boundary_condition(const std::string &bcface) const;
			bool is_PML(const std::string &bc) const { return (bc == "PML") ? true : false; };
			bool is_data_consistent(const Godzilla::vecxd &data1, const Godzilla::vecxd &data2,
									const Godzilla::vecxd &data3, const Godzilla::vecxd &data4) const;
	};
}