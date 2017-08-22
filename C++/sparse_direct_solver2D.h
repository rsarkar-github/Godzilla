#pragma once
#include "boundary_condition2D.h"
#include "velocity2D.h"
#include "field2D.h"

namespace Godzilla {
	namespace Helmholtz2DReal {
		class SparseDirectSolver2D {
			public:
				// Constructors
				SparseDirectSolver2D();

				// Public methods

			private:
				// Private members
				// Only the following are user dependent
				Godzilla::Velocity2D _vel2D;
				Godzilla::Field2D _forcing2D;
				Godzilla::BoundaryCondition2D _bc2D;

				// The rest are initialized from the user inputs
				// Geometry for simulation
				Godzilla::Geometry2D _sim_geom2D;

				// Padding cells for left, top, right and bottom faces of input geometry
				// with respect to simulation geometry
				size_t _pad1;
				size_t _pad2;
				size_t _pad3;
				size_t _pad4;

				// Boundary conditions data for left, top, right and bottom faces of input geometry
				// with respect to simulation geometry
				Godzilla::vecxd _sim_bc_face1;
				Godzilla::vecxd _sim_bc_face2;
				Godzilla::vecxd _sim_bc_face3;
				Godzilla::vecxd _sim_bc_face4;

				// Private methods
				size_t calculate_pad1(const Godzilla::BoundaryCondition2D &bc2D) const;
				size_t calculate_pad2(const Godzilla::BoundaryCondition2D &bc2D) const;
				size_t calculate_pad3(const Godzilla::BoundaryCondition2D &bc2D) const;
				size_t calculate_pad4(const Godzilla::BoundaryCondition2D &bc2D) const;
				Godzilla::Geometry2D create_sim_geom2D(const Godzilla::BoundaryCondition2D &bc2D) const;
				
				Godzilla::vecxd field_simgrid_zeropad(const Godzilla::Field2D &field2D_in, const Godzilla::Geometry2D &geom2D_out,
													  const size_t &pad1, const size_t &pad4) const;

				Godzilla::vecxd velocity_simgrid_extrap(const Godzilla::Velocity2D &vel2D_in, const Godzilla::Geometry2D &geom2D_out,
														const size_t &pad1, const size_t &pad2, const size_t &pad3, const size_t &pad4) const;

				Godzilla::vecxd calculate_sim_bc_face1(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
													   const size_t &pad4, const size_t &pad2) const;
				Godzilla::vecxd calculate_sim_bc_face2(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
													   const size_t &pad1, const size_t &pad3) const;
				Godzilla::vecxd calculate_sim_bc_face3(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
													   const size_t &pad4, const size_t &pad2) const;
				Godzilla::vecxd calculate_sim_bc_face4(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
													   const size_t &pad1, const size_t &pad3) const;
		};
	}
}