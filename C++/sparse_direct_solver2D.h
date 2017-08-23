#pragma once
#include "boundary_condition2D.h"
#include "velocity2D.h"
#include "field2D.h"
#include "umfpack.h"

namespace Godzilla {
	namespace Helmholtz2DReal {
		class SparseDirectSolver2D {
			public:
				// Constructors
				SparseDirectSolver2D() = delete;
				SparseDirectSolver2D(const Godzilla::Velocity2D &vel2D, const Godzilla::Field2D &forcing2D,
									 const Godzilla::BoundaryCondition2D &bc2D, const double &omega);

				// Public methods
				void change_velocity_data(const Godzilla::Velocity2D &vel2D);
				void change_forcing_data(const Godzilla::Field2D &forcing2D);
				void change_boundary_conditions(const Godzilla::BoundaryCondition2D &bc2D);
				void change_omega(const double &omega);
				void create_sparse_matrix();

			private:
				// Private members
				// Only the following are user dependent
				const Godzilla::Velocity2D *_vel2D;
				const Godzilla::Field2D *_forcing2D;
				const Godzilla::BoundaryCondition2D *_bc2D;
				double _omega;

				// The rest are initialized from the user inputs
				// If initialization was successful
				bool _initialized_state;

				// If solver is ready to solve
				bool _is_matrix_ready;
				bool _is_rhs_ready;

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

				// Velocity and forcing data extrapolated and zeropadded to siimulation grid respectively
				Godzilla::vecxd _velocity_data_simgrid;
				Godzilla::vecxd _forcing_data_simgrid;

				// Private methods
				bool is_velocity_real(const Godzilla::Velocity2D &vel2D_in) const;
				bool is_solver_ready() const { return (_is_matrix_ready && _is_rhs_ready); }

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

				Godzilla::vecxd calculate_sX(const Godzilla::vecd &x, const Godzilla::Geometry2D &geom2D, const double &omega,
											 const size_t &pad1, const size_t &pad3) const;
				Godzilla::vecxd calculate_sY(const Godzilla::vecd &y, const Godzilla::Geometry2D &geom2D, const double &omega,
											 const size_t &pad4, const size_t &pad2) const;
		};
	}
}