#pragma once
#include "lock_manager.h"
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
				SparseDirectSolver2D(const Godzilla::Velocity2D *vel2D, const Godzilla::Field2D *forcing2D,
									 const Godzilla::BoundaryCondition2D *bc2D, const double &omega, const int &stencil_type = 0);

				// Public methods
				void change_velocity_data(const Godzilla::Velocity2D *vel2D);
				void change_forcing_data(const Godzilla::Field2D *forcing2D);
				void change_boundary_conditions(const Godzilla::BoundaryCondition2D *bc2D);
				void change_omega(const double &omega);
				void change_stencil_type(const int &stencil_type);
				void create_sparse_matrix_rhs();
				void solve();
				void extract_solution(Godzilla::Field2D &solution2D) const;
				void free_solver_resources();

				void print_matrix(const double &umfpack_prl) const;
				void print_rhs() const;
				void print_solution() const;

			private:
				// Private members

				// Only the following are user dependent

				const Godzilla::Velocity2D *_vel2D;
				const Godzilla::Field2D *_forcing2D;
				const Godzilla::BoundaryCondition2D *_bc2D;
				double _omega;

				// Stencil types
				// stencil_type = 0 : symmetric 5 point stencil (centered difference)
				int _stencil_type;

				// The rest are initialized from the user inputs

				// _initialized_state : if initialization was successful
				// _is_matrix_ready : if solver is ready to solve
				// _is_rhs_ready : if solver is ready to solve
				// _sim_geom2D : geometry for simulation
				// _pad1 : number of padded cells on left face of input geometry
				// _pad2 : number of padded cells on top face of input geometry
				// _pad3 : number of padded cells on right face of input geometry
				// _pad4 : number of padded cells on bottom face of input geometry
				// _sim_bc_face1 : boundary conditions for left face of input geometry
				// _sim_bc_face2 : boundary conditions for top face of input geometry
				// _sim_bc_face3 : boundary conditions for right face of input geometry
				// _sim_bc_face4 : boundary conditions for bottom face of input geometry
				// _velocity_data_simgrid : velocity data extrapolated to simulation grid
				// _forcing_data_simgrid : forcing data zeropadded to simulation grid

				bool _initialized_state;
				bool _is_matrix_ready;
				bool _is_rhs_ready;
				Godzilla::Geometry2D _sim_geom2D;
				size_t _pad1;
				size_t _pad2;
				size_t _pad3;
				size_t _pad4;
				Godzilla::vecxd _sim_bc_face1;
				Godzilla::vecxd _sim_bc_face2;
				Godzilla::vecxd _sim_bc_face3;
				Godzilla::vecxd _sim_bc_face4;
				Godzilla::vecxd _velocity_data_simgrid;
				Godzilla::vecxd _forcing_data_simgrid;

				// UMFPACK objects for solving Ay = b
				// _dim_A : dimensions of A
				// *_A_p : indices of start position for first non-zero entry of each column
				// *_A_i : row indices of entries of A
				// *_A_x : real part of entries of A
				// *_A_z : complex part of entries of A
				// *_b_x : real part of b
				// *_b_z : complex part of b
				// *_y_x : real part of y
				// *_y_z : complex part of y
				// *_symbolic : needed by UMFPACK for symbolic factorization
				// *_numeric : needed by UMFPACK for numeric factorization

				SuiteSparse_long _dim_A;
				SuiteSparse_long *_A_p;
				SuiteSparse_long *_A_i;
				double *_A_x;
				double *_A_z;
				double *_b_x;
				double *_b_z;
				double *_y_x;
				double *_y_z;
				void *_symbolic;
				void *_numeric;

				//***********************************************************************************************
				// Private methods
				bool is_velocity_real(const Godzilla::Velocity2D &vel2D_in) const;
				bool is_valid_stencil(const int &stencil_type) const;
				bool is_solver_ready() const { return (_is_matrix_ready && _is_rhs_ready); }
				Godzilla::Geometry2D create_sim_geom2D(const Godzilla::BoundaryCondition2D &bc2D) const;
				size_t calculate_pad1(const Godzilla::BoundaryCondition2D &bc2D) const;
				size_t calculate_pad2(const Godzilla::BoundaryCondition2D &bc2D) const;
				size_t calculate_pad3(const Godzilla::BoundaryCondition2D &bc2D) const;
				size_t calculate_pad4(const Godzilla::BoundaryCondition2D &bc2D) const;
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
				void append_A(std::vector<size_t> &row_A, std::vector<size_t> &col_A, Godzilla::vecxd &val_A,
							  const size_t &row, const size_t &col, const Godzilla::xd &val) const;
				void append_b(std::vector<size_t> &row_b, Godzilla::vecxd &val_b, const size_t &row, const Godzilla::xd &val) const;
				void set_A(const SuiteSparse_long &n, SuiteSparse_long *A_p, SuiteSparse_long *A_i, double *A_x, double *A_z,
						   const std::vector<size_t> &row_A, const std::vector<size_t> &col_A, const Godzilla::vecxd &val_A);
				void set_b(const size_t &n, double *b_x, double *b_z, const std::vector<size_t> &row_b, const Godzilla::vecxd &val_b);
		};
	}
}