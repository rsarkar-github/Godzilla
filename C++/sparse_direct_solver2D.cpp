#include "sparse_direct_solver2D.h"
#include <cmath>
#include <algorithm>
#include <cassert>

namespace Godzilla {
	namespace Helmholtz2DReal {
		// Constructors
		SparseDirectSolver2D::SparseDirectSolver2D(const Godzilla::Velocity2D *vel2D, const Godzilla::Field2D *forcing2D,
												   const Godzilla::BoundaryCondition2D *bc2D, const double &omega, const int &stencil_type)

			: _vel2D(nullptr), _forcing2D(nullptr), _bc2D(nullptr), _initialized_state(false), _is_matrix_ready(false), _is_rhs_ready(false),
			  _dim_A(0), _A_p(nullptr), _A_i(nullptr), _A_x(nullptr), _A_z(nullptr), _b_x(nullptr), _b_z(nullptr), _y_x(nullptr), _y_z(nullptr),
			  _symbolic(nullptr), _numeric(nullptr) {

			_omega = (omega != 0.) ? omega : 1.;
			_stencil_type = (this->is_valid_stencil(stencil_type)) ? stencil_type : 0;

			const Godzilla::Geometry2D &geom2D = vel2D->get_geom2D();
			if (this->is_velocity_real(*vel2D) && geom2D.is_equal(forcing2D->get_geom2D()) && geom2D.is_equal(bc2D->get_geom2D())) {
				_vel2D = vel2D;
				_forcing2D = forcing2D;
				_bc2D = bc2D;
				_sim_geom2D = this->create_sim_geom2D(*_bc2D);
				_pad1 = this->calculate_pad1(*_bc2D);
				_pad2 = this->calculate_pad2(*_bc2D);
				_pad3 = this->calculate_pad3(*_bc2D);
				_pad4 = this->calculate_pad4(*_bc2D);
				_sim_bc_face1 = this->calculate_sim_bc_face1(*_bc2D, _sim_geom2D, _pad4, _pad2);
				_sim_bc_face2 = this->calculate_sim_bc_face2(*_bc2D, _sim_geom2D, _pad1, _pad3);
				_sim_bc_face3 = this->calculate_sim_bc_face3(*_bc2D, _sim_geom2D, _pad4, _pad2);
				_sim_bc_face4 = this->calculate_sim_bc_face4(*_bc2D, _sim_geom2D, _pad1, _pad3);
				_velocity_data_simgrid = this->velocity_simgrid_extrap(*_vel2D, _sim_geom2D, _pad1, _pad2, _pad3, _pad4);
				_forcing_data_simgrid = this->field_simgrid_zeropad(*_forcing2D, _sim_geom2D, _pad1, _pad4);
				_initialized_state = true;
			}
			else {
				std::cerr << "Velocity not real / Geometries did not match. Initialization failed." << std::endl;
			}
		}

		// Public methods
		void SparseDirectSolver2D::change_velocity_data(const Godzilla::Velocity2D *vel2D) {
			if (_initialized_state) {
				if (_vel2D->get_geom2D().is_equal(vel2D->get_geom2D()) && this->is_velocity_real(*vel2D)) {
						_vel2D = vel2D;
						_velocity_data_simgrid = this->velocity_simgrid_extrap(*_vel2D, _sim_geom2D, _pad1, _pad2, _pad3, _pad4);
						
						_is_matrix_ready = false;
						_dim_A = 0;
						delete[] _A_p;
						delete[] _A_i;
						delete[] _A_x;
						delete[] _A_z;
						delete[] _y_x;
						delete[] _y_z;
						umfpack_zl_free_numeric(&_numeric);
						_A_p = nullptr;
						_A_i = nullptr;
						_A_x = nullptr;
						_A_z = nullptr;
						_y_x = nullptr;
						_y_z = nullptr;
						_numeric = nullptr;
				}
				else {
					std::cerr << "Input geometry does not match existing geometry or velocity not real. Cannot change velocity data." << std::endl;
				}
			}
			else {
				std::cerr << "Object not initialized. Cannot change velocity data. Initialize object first." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_forcing_data(const Godzilla::Field2D *forcing2D) {
			if (_initialized_state) {
				if (_forcing2D->get_geom2D().is_equal(forcing2D->get_geom2D())) {
					_forcing2D = forcing2D;
					_forcing_data_simgrid = this->field_simgrid_zeropad(*_forcing2D, _sim_geom2D, _pad1, _pad4);
						
					/*_is_rhs_ready = false;
					delete[] _y_x;
					delete[] _y_z;
					delete[] _b_x;
					delete[] _b_z;
					_y_x = nullptr;
					_y_z = nullptr;
					_b_x = nullptr;
					_b_z = nullptr;*/
					_is_matrix_ready = false;
					_is_rhs_ready = false;
					_dim_A = 0;
					delete[] _A_p;
					delete[] _A_i;
					delete[] _A_x;
					delete[] _A_z;
					delete[] _y_x;
					delete[] _y_z;
					delete[] _b_x;
					delete[] _b_z;
					umfpack_zl_free_numeric(&_numeric);
					_A_p = nullptr;
					_A_i = nullptr;
					_A_x = nullptr;
					_A_z = nullptr;
					_y_x = nullptr;
					_y_z = nullptr;
					_b_x = nullptr;
					_b_z = nullptr;
					_numeric = nullptr;
				}
				else {
					std::cerr << "Input geometry does not match existing geometry. Cannot change forcing data." << std::endl;
				}
			}
			else {
				std::cerr << "Object not initialized. Cannot change forcing data. Initialize object first." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_boundary_conditions(const Godzilla::BoundaryCondition2D *bc2D) {
			if (_initialized_state) {
				if (_vel2D->get_geom2D().is_equal(bc2D->get_geom2D())) {
					_bc2D = bc2D;
					_sim_geom2D = this->create_sim_geom2D(*_bc2D);
					_pad1 = this->calculate_pad1(*_bc2D);
					_pad2 = this->calculate_pad2(*_bc2D);
					_pad3 = this->calculate_pad3(*_bc2D);
					_pad4 = this->calculate_pad4(*_bc2D);
					_sim_bc_face1 = this->calculate_sim_bc_face1(*_bc2D, _sim_geom2D, _pad4, _pad2);
					_sim_bc_face2 = this->calculate_sim_bc_face2(*_bc2D, _sim_geom2D, _pad1, _pad3);
					_sim_bc_face3 = this->calculate_sim_bc_face3(*_bc2D, _sim_geom2D, _pad4, _pad2);
					_sim_bc_face4 = this->calculate_sim_bc_face4(*_bc2D, _sim_geom2D, _pad1, _pad3);
					_velocity_data_simgrid = this->velocity_simgrid_extrap(*_vel2D, _sim_geom2D, _pad1, _pad2, _pad3, _pad4);
					_forcing_data_simgrid = this->field_simgrid_zeropad(*_forcing2D, _sim_geom2D, _pad1, _pad4);
					
					_is_matrix_ready = false;
					_is_rhs_ready = false;
					_dim_A = 0;
					delete[] _A_p;
					delete[] _A_i;
					delete[] _A_x;
					delete[] _A_z;
					delete[] _y_x;
					delete[] _y_z;
					delete[] _b_x;
					delete[] _b_z;
					umfpack_zl_free_symbolic(&_symbolic);
					umfpack_zl_free_numeric(&_numeric);
					_A_p = nullptr;
					_A_i = nullptr;
					_A_x = nullptr;
					_A_z = nullptr;
					_y_x = nullptr;
					_y_z = nullptr;
					_b_x = nullptr;
					_b_z = nullptr;
					_symbolic = nullptr;
					_numeric = nullptr;
				}
				else {
					std::cerr << "Input geometry does not match existing geometry. Cannot change boundary conditions." << std::endl;
				}
			}
			else {
				std::cerr << "Object not initialized. Cannot change boundary conditions. Initialize object first." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_omega(const double &omega) {
			if ((omega != 0.) && (omega != _omega)) {
				_omega = omega;
				
				_is_matrix_ready = false;
				_is_rhs_ready = false;
				_dim_A = 0;
				delete[] _A_p;
				delete[] _A_i;
				delete[] _A_x;
				delete[] _A_z;
				delete[] _y_x;
				delete[] _y_z;
				delete[] _b_x;
				delete[] _b_z;
				umfpack_zl_free_numeric(&_numeric);
				_A_p = nullptr;
				_A_i = nullptr;
				_A_x = nullptr;
				_A_z = nullptr;
				_y_x = nullptr;
				_y_z = nullptr;
				_b_x = nullptr;
				_b_z = nullptr;
				_numeric = nullptr;
			}
			else {
				std::cerr << "Omega unchanged. Either omega equals 0 or same as existing omega." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_stencil_type(const int &stencil_type) {
			if ((stencil_type != _stencil_type) && (this->is_valid_stencil(stencil_type))) {
				_stencil_type = stencil_type;

				_is_matrix_ready = false;
				_is_rhs_ready = false;
				_dim_A = 0;
				delete[] _A_p;
				delete[] _A_i;
				delete[] _A_x;
				delete[] _A_z;
				delete[] _y_x;
				delete[] _y_z;
				delete[] _b_x;
				delete[] _b_z;
				umfpack_zl_free_symbolic(&_symbolic);
				umfpack_zl_free_numeric(&_numeric);
				_A_p = nullptr;
				_A_i = nullptr;
				_A_x = nullptr;
				_A_z = nullptr;
				_y_x = nullptr;
				_y_z = nullptr;
				_b_x = nullptr;
				_b_z = nullptr;
				_symbolic = nullptr;
				_numeric = nullptr;
			}
			else {
				std::cerr << "Stencil type unchanged. Either not a valid stencil type or same as existing stencil." << std::endl;
			}
		}

		void SparseDirectSolver2D::create_sparse_matrix_rhs() {
			
			// Check if object is in initialized state
			if (!_initialized_state) {
				std::cerr << "Solver not yet initialized." << std::endl;
				return;
			}

			// Check if solver is ready : matrix and rhs are already created
			if (this->is_solver_ready()) {
				std::cerr << "Solver already ready. Matrix and rhs have been created." << std::endl;
				return;
			}

			// If solver is ready : create matrix and rhs

			// Get nX, nY, startX, startY, hX, and hY in simulation geometry
			// Get start and end point indices along X and Y directions (start_nX, end_nX, start_nY, end_nY)
			// Calculate number of active points in each direction and total active grid points to solve for
			// Throw error if no active grid points (put some thought later on how to handle this)
			const size_t nX = _sim_geom2D.get_nX();
			const size_t nY = _sim_geom2D.get_nY();
			const double startX = _sim_geom2D.get_startX();
			const double startY = _sim_geom2D.get_startY();
			const double hX = std::abs(_sim_geom2D.get_hX());
			const double hY = std::abs(_sim_geom2D.get_hY());
			const size_t start_nX = (_bc2D->get_bc_face1() != "NBC") ? 1 : 0;
			const size_t end_nX = (_bc2D->get_bc_face3() != "NBC") ? (_sim_geom2D.get_nX() - 2) : (_sim_geom2D.get_nX() - 1);
			const size_t start_nY = (_bc2D->get_bc_face4() != "NBC") ? 1 : 0;
			const size_t end_nY = (_bc2D->get_bc_face2() != "NBC") ? (_sim_geom2D.get_nY() - 2) : (_sim_geom2D.get_nY() - 1);
			const size_t points_X = (start_nX <= end_nX) ? (end_nX - start_nX) + 1 : 0;
			const size_t points_Y = (start_nY <= end_nY) ? (end_nY - start_nY) + 1 : 0;

			const size_t active_points = points_X * points_Y;
			if (active_points == 0) {
				std::cerr << "No active points to solve for. Boundary conditions determine solution." << std::endl;
				return;
			}

			// Define vectors to create sparse matrix in triplet form
			std::vector<size_t> row_A, col_A, row_b;
			Godzilla::vecxd val_A, val_b;
			row_A.reserve(active_points * 5);
			col_A.reserve(active_points * 5);
			val_A.reserve(active_points * 5);
			row_b.reserve(active_points);
			val_b.reserve(active_points);
			
			/////////////////////////////////////////////////////////////////////////////////////////////
			// STENCIL TYPE = 0
			if (_stencil_type == 0) {

				// Calculate sX and sY
				const size_t sX_points = 1 + 2 * _sim_geom2D.get_ncellsX();
				const size_t sY_points = 1 + 2 * _sim_geom2D.get_ncellsY();
				Godzilla::vecd x(sX_points, 0.);
				Godzilla::vecd y(sY_points, 0.);
				double *ptr = x.data();
				double stepX = hX / 2.;
				for (size_t i = 0; i < sX_points; ++i) {
					ptr[i] = startX + i * stepX;
				}
				ptr = y.data();
				double stepY = hY / 2.;
				for (size_t i = 0; i < sY_points; ++i) {
					ptr[i] = startY + i * stepY;
				}
				Godzilla::vecxd sX = this->calculate_sX(x, _sim_geom2D, _omega, _pad1, _pad3);
				Godzilla::vecxd sY = this->calculate_sY(y, _sim_geom2D, _omega, _pad4, _pad2);
				for (auto &i : sX) {
					i /= hX;
				}
				for (auto &i : sY) {
					i /= hY;
				}

				// Get pointers to the sX, sY, velocity, forcing and boundary conditions along the 4 faces
				const Godzilla::xd *ptr_sX = sX.data();
				const Godzilla::xd *ptr_sY = sY.data();
				const Godzilla::xd *ptr_velocity2D = _velocity_data_simgrid.data();
				const Godzilla::xd *ptr_forcing2D = _forcing_data_simgrid.data();
				const Godzilla::xd *ptr_bc_face1 = _sim_bc_face1.data();
				const Godzilla::xd *ptr_bc_face2 = _sim_bc_face2.data();
				const Godzilla::xd *ptr_bc_face3 = _sim_bc_face3.data();
				const Godzilla::xd *ptr_bc_face4 = _sim_bc_face4.data();

				// Define variables that will be used
				size_t n_index = 0;
				size_t sX_index = 0;
				size_t sY_index = 0;

				size_t velocity_index = 0;
				size_t forcing_index = 0;

				size_t bc_face1_index = 0;
				size_t bc_face2_index = 0;
				size_t bc_face3_index = 0;
				size_t bc_face4_index = 0;

				Godzilla::xd f(_omega, 0.);
				Godzilla::xd val(0., 0.);

				Godzilla::xd p1X(0., 0.);
				Godzilla::xd p2X(0., 0.);
				Godzilla::xd p3X(0., 0.);
				Godzilla::xd p1Y(0., 0.);
				Godzilla::xd p2Y(0., 0.);
				Godzilla::xd p3Y(0., 0.);

				// Handle interior points
				for (size_t i = 1; i < points_Y - 1; ++i) {

					n_index = i * points_X + 1;
					velocity_index = (start_nY + i) * nX + start_nX + 1;
					forcing_index = velocity_index;

					sY_index = 2 * (start_nY + i);
					sX_index = 2 * (start_nX + 1);

					p1Y = ptr_sY[sY_index];
					p2Y = ptr_sY[sY_index + 1];
					p3Y = ptr_sY[sY_index - 1];

					for (size_t j = 1; j < points_X - 1; ++j) {

						p1X = ptr_sX[sX_index];
						p2X = ptr_sX[sX_index + 1];
						p3X = ptr_sX[sX_index - 1];

						// Forcing term
						val = ptr_forcing2D[forcing_index];
						this->append_b(row_b, val_b, n_index, val);

						// Central coefficient
						val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
						this->append_A(row_A, col_A, val_A, n_index, n_index, val);

						// Left coefficient
						val = p1X * p3X;
						this->append_A(row_A, col_A, val_A, n_index, n_index - 1, val);

						// Right coefficient
						val = p1X * p2X;
						this->append_A(row_A, col_A, val_A, n_index, n_index + 1, val);

						// Bottom coefficient
						val = p1Y * p3Y;
						this->append_A(row_A, col_A, val_A, n_index, n_index - points_X, val);

						// Top coefficient
						val = p1Y * p2Y;
						this->append_A(row_A, col_A, val_A, n_index, n_index + points_X, val);

						sX_index += 2;
						++n_index;
						++velocity_index;
						++forcing_index;
					}
					sY_index += 2;
				}

				// Handle face1 except corners
				n_index = points_X;
				velocity_index = (start_nY + 1) * nX + start_nX;
				forcing_index = velocity_index;
				bc_face1_index = start_nY + 1;

				sX_index = 2 * start_nX;
				sY_index = 2 * (start_nY + 1);

				p1X = ptr_sX[sX_index];
				p2X = ptr_sX[sX_index + 1];
				p3X = ptr_sX[sX_index - 1];

				for (size_t i = 1; i < points_Y - 1; ++i) {
					p1Y = ptr_sY[sY_index];
					p2Y = ptr_sY[sY_index + 1];
					p3Y = ptr_sY[sY_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face1() != "NBC") ? p1X * p3X * ptr_bc_face1[bc_face1_index] : 2 * hX * p1X * p3X * ptr_bc_face1[bc_face1_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Right coefficient
					val = (_bc2D->get_bc_face1() != "NBC") ? p1X * p2X : p1X * (p2X + p3X);
					this->append_A(row_A, col_A, val_A, n_index, n_index + 1, val);

					// Bottom coefficient
					val = p1Y * p3Y;
					this->append_A(row_A, col_A, val_A, n_index, n_index - points_X, val);

					// Top coefficient
					val = p1Y * p2Y;
					this->append_A(row_A, col_A, val_A, n_index, n_index + points_X, val);

					sY_index += 2;
					n_index += points_X;
					velocity_index += nX;
					forcing_index += nX;
					++bc_face1_index;
				}

				// Handle face2 except corners
				n_index = (points_Y - 1) * points_X + 1;
				velocity_index = end_nY * nX + start_nX + 1;
				forcing_index = velocity_index;
				bc_face2_index = start_nX + 1;

				sX_index = 2 * (start_nX + 1);
				sY_index = 2 * end_nY;

				p1Y = ptr_sY[sY_index];
				p2Y = ptr_sY[sY_index + 1];
				p3Y = ptr_sY[sY_index - 1];

				for (size_t i = 1; i < points_X - 1; ++i) {

					p1X = ptr_sX[sX_index];
					p2X = ptr_sX[sX_index + 1];
					p3X = ptr_sX[sX_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face2() != "NBC") ? p1Y * p2Y * ptr_bc_face2[bc_face2_index] : 2 * hY * p1Y * p2Y * ptr_bc_face2[bc_face2_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Left coefficient
					val = p1X * p3X;
					this->append_A(row_A, col_A, val_A, n_index, n_index - 1, val);

					// Right coefficient
					val = p1X * p2X;
					this->append_A(row_A, col_A, val_A, n_index, n_index + 1, val);

					// Bottom coefficient
					val = (_bc2D->get_bc_face2() != "NBC") ? p1Y * p3Y : p1Y * (p2Y + p3Y);
					this->append_A(row_A, col_A, val_A, n_index, n_index - points_X, val);

					sX_index += 2;
					++n_index;
					++velocity_index;
					++forcing_index;
					++bc_face2_index;
				}

				// Handle face3 except corners
				n_index = 2 * points_X - 1;
				velocity_index = (start_nY + 1) * nX + end_nX;
				forcing_index = velocity_index;
				bc_face3_index = start_nY + 1;

				sX_index = 2 * end_nX;
				sY_index = 2 * (start_nY + 1);

				p1X = ptr_sX[sX_index];
				p2X = ptr_sX[sX_index + 1];
				p3X = ptr_sX[sX_index - 1];

				for (size_t i = 1; i < points_Y - 1; ++i) {

					p1Y = ptr_sY[sY_index];
					p2Y = ptr_sY[sY_index + 1];
					p3Y = ptr_sY[sY_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face3() != "NBC") ? p1X * p2X * ptr_bc_face3[bc_face3_index] : 2 * hX * p1X * p2X * ptr_bc_face3[bc_face3_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Left coefficient
					val = (_bc2D->get_bc_face3() != "NBC") ? p1X * p3X : p1X * (p2X + p3X);
					this->append_A(row_A, col_A, val_A, n_index, n_index - 1, val);

					// Bottom coefficient
					val = p1Y * p3Y;
					this->append_A(row_A, col_A, val_A, n_index, n_index - points_X, val);

					// Top coefficient
					val = p1Y * p2Y;
					this->append_A(row_A, col_A, val_A, n_index, n_index + points_X, val);

					sY_index += 2;
					n_index += points_X;
					velocity_index += nX;
					forcing_index += nX;
					++bc_face3_index;
				}

				// Handle face4 except corners
				n_index = 1;
				velocity_index = start_nY * nX + start_nX + 1;
				forcing_index = velocity_index;
				bc_face4_index = start_nX + 1;

				sX_index = 2 * (start_nX + 1);
				sY_index = 2 * start_nY;

				p1Y = ptr_sY[sY_index];
				p2Y = ptr_sY[sY_index + 1];
				p3Y = ptr_sY[sY_index - 1];

				for (size_t i = 1; i < points_X - 1; ++i) {

					p1X = ptr_sX[sX_index];
					p2X = ptr_sX[sX_index + 1];
					p3X = ptr_sX[sX_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face4() != "NBC") ? p1Y * p3Y * ptr_bc_face4[bc_face4_index] : 2 * hY * p1Y * p3Y * ptr_bc_face4[bc_face4_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Left coefficient
					val = p1X * p3X;
					this->append_A(row_A, col_A, val_A, n_index, n_index - 1, val);

					// Right coefficient
					val = p1X * p2X;
					this->append_A(row_A, col_A, val_A, n_index, n_index + 1, val);

					// Top coefficient
					val = (_bc2D->get_bc_face4() != "NBC") ? p1Y * p2Y : p1Y * (p2Y + p3Y);
					this->append_A(row_A, col_A, val_A, n_index, n_index + points_X, val);

					sX_index += 2;
					++n_index;
					++velocity_index;
					++forcing_index;
					++bc_face4_index;
				}

				if ((points_X > 1) && (points_Y > 1)) {
					// Handle face1-face2 corners (top left)
					n_index = (points_Y - 1) * points_X;
					velocity_index = end_nY * nX + start_nX;
					forcing_index = velocity_index;
					bc_face1_index = end_nY;
					bc_face2_index = start_nX;

					sX_index = 2 * start_nX;
					sY_index = 2 * end_nY;

					p1Y = ptr_sY[sY_index];
					p2Y = ptr_sY[sY_index + 1];
					p3Y = ptr_sY[sY_index - 1];
					p1X = ptr_sX[sX_index];
					p2X = ptr_sX[sX_index + 1];
					p3X = ptr_sX[sX_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face1() != "NBC") ? p1X * p3X * ptr_bc_face1[bc_face1_index] : 2 * hX * p1X * p3X * ptr_bc_face1[bc_face1_index];
					val -= (_bc2D->get_bc_face2() != "NBC") ? p1Y * p2Y * ptr_bc_face2[bc_face2_index] : 2 * hY * p1Y * p2Y * ptr_bc_face2[bc_face2_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Right coefficient
					val = (_bc2D->get_bc_face1() != "NBC") ? p1X * p2X : p1X * (p2X + p3X);
					this->append_A(row_A, col_A, val_A, n_index, n_index + 1, val);

					// Bottom coefficient
					val = (_bc2D->get_bc_face2() != "NBC") ? p1Y * p3Y : p1Y * (p2Y + p3Y);
					this->append_A(row_A, col_A, val_A, n_index, n_index - points_X, val);

					// Handle face2-face3 corners (top right)
					n_index = points_Y * points_X - 1;
					velocity_index = end_nY * nX + end_nX;
					forcing_index = velocity_index;
					bc_face2_index = end_nX;
					bc_face3_index = end_nY;

					sX_index = 2 * end_nX;
					sY_index = 2 * end_nY;

					p1Y = ptr_sY[sY_index];
					p2Y = ptr_sY[sY_index + 1];
					p3Y = ptr_sY[sY_index - 1];
					p1X = ptr_sX[sX_index];
					p2X = ptr_sX[sX_index + 1];
					p3X = ptr_sX[sX_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face2() != "NBC") ? p1Y * p2Y * ptr_bc_face2[bc_face2_index] : 2 * hY * p1Y * p2Y * ptr_bc_face2[bc_face2_index];
					val -= (_bc2D->get_bc_face3() != "NBC") ? p1X * p2X * ptr_bc_face3[bc_face3_index] : 2 * hX * p1X * p2X * ptr_bc_face3[bc_face3_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Left coefficient
					val = (_bc2D->get_bc_face3() != "NBC") ? p1X * p3X : p1X * (p2X + p3X);
					this->append_A(row_A, col_A, val_A, n_index, n_index - 1, val);

					// Bottom coefficient
					val = (_bc2D->get_bc_face2() != "NBC") ? p1Y * p3Y : p1Y * (p2Y + p3Y);
					this->append_A(row_A, col_A, val_A, n_index, n_index - points_X, val);

					// Handle face3-face4 corners (bottom right)
					n_index = points_X - 1;
					velocity_index = start_nY * nX + end_nX;
					forcing_index = velocity_index;
					bc_face3_index = start_nY;
					bc_face4_index = end_nX;

					sX_index = 2 * end_nX;
					sY_index = 2 * start_nY;

					p1Y = ptr_sY[sY_index];
					p2Y = ptr_sY[sY_index + 1];
					p3Y = ptr_sY[sY_index - 1];
					p1X = ptr_sX[sX_index];
					p2X = ptr_sX[sX_index + 1];
					p3X = ptr_sX[sX_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face3() != "NBC") ? p1X * p2X * ptr_bc_face3[bc_face3_index] : 2 * hX * p1X * p2X * ptr_bc_face3[bc_face3_index];
					val -= (_bc2D->get_bc_face4() != "NBC") ? p1Y * p3Y * ptr_bc_face4[bc_face4_index] : 2 * hY * p1Y * p3Y * ptr_bc_face4[bc_face4_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Left coefficient
					val = (_bc2D->get_bc_face3() != "NBC") ? p1X * p3X : p1X * (p2X + p3X);
					this->append_A(row_A, col_A, val_A, n_index, n_index - 1, val);

					// Top coefficient
					val = (_bc2D->get_bc_face4() != "NBC") ? p1Y * p2Y : p1Y * (p2Y + p3Y);
					this->append_A(row_A, col_A, val_A, n_index, n_index + points_X, val);

					// Handle face4-face1 corners (bottom left)
					n_index = 0;
					velocity_index = start_nY * nX + start_nX;
					forcing_index = velocity_index;
					bc_face4_index = start_nX;
					bc_face1_index = start_nY;

					sX_index = 2 * start_nX;
					sY_index = 2 * start_nY;

					p1Y = ptr_sY[sY_index];
					p2Y = ptr_sY[sY_index + 1];
					p3Y = ptr_sY[sY_index - 1];
					p1X = ptr_sX[sX_index];
					p2X = ptr_sX[sX_index + 1];
					p3X = ptr_sX[sX_index - 1];

					// Forcing term
					val = ptr_forcing2D[forcing_index];
					val -= (_bc2D->get_bc_face4() != "NBC") ? p1Y * p3Y * ptr_bc_face4[bc_face4_index] : 2 * hY * p1Y * p3Y * ptr_bc_face4[bc_face4_index];
					val -= (_bc2D->get_bc_face1() != "NBC") ? p1X * p3X * ptr_bc_face1[bc_face1_index] : 2 * hX * p1X * p3X * ptr_bc_face1[bc_face1_index];
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Top coefficient
					val = (_bc2D->get_bc_face4() != "NBC") ? p1Y * p2Y : p1Y * (p2Y + p3Y);
					this->append_A(row_A, col_A, val_A, n_index, n_index + points_X, val);

					// Right coefficient
					val = (_bc2D->get_bc_face1() != "NBC") ? p1X * p2X : p1X * (p2X + p3X);
					this->append_A(row_A, col_A, val_A, n_index, n_index + 1, val);
				}
				else {
					if ((points_X == 1) && (points_Y == 1)) {
						n_index = 0;
						velocity_index = start_nY * nX + start_nX;
						forcing_index = velocity_index;
						bc_face1_index = start_nY;
						bc_face2_index = start_nX;
						bc_face3_index = start_nY;
						bc_face4_index = start_nX;

						sX_index = 2 * start_nX;
						sY_index = 2 * start_nY;

						p1Y = ptr_sY[sY_index];
						p2Y = ptr_sY[sY_index + 1];
						p3Y = ptr_sY[sY_index - 1];
						p1X = ptr_sX[sX_index];
						p2X = ptr_sX[sX_index + 1];
						p3X = ptr_sX[sX_index - 1];

						// Forcing term
						val = ptr_forcing2D[forcing_index];
						val -= (_bc2D->get_bc_face1() != "NBC") ? p1X * p3X * ptr_bc_face1[bc_face1_index] : 2 * hX * p1X * p3X * ptr_bc_face1[bc_face1_index];
						val -= (_bc2D->get_bc_face2() != "NBC") ? p1Y * p2Y * ptr_bc_face2[bc_face2_index] : 2 * hY * p1Y * p2Y * ptr_bc_face2[bc_face2_index];
						val -= (_bc2D->get_bc_face3() != "NBC") ? p1X * p2X * ptr_bc_face3[bc_face3_index] : 2 * hX * p1X * p2X * ptr_bc_face3[bc_face3_index];
						val -= (_bc2D->get_bc_face4() != "NBC") ? p1Y * p3Y * ptr_bc_face4[bc_face4_index] : 2 * hY * p1Y * p3Y * ptr_bc_face4[bc_face4_index];
						this->append_b(row_b, val_b, n_index, val);

						// Central coefficient
						val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
						this->append_A(row_A, col_A, val_A, n_index, n_index, val);
					}
					else {
						std::cerr << "Either (points_X == 1) or (points_Y == 1). Case not supported currectly." << std::endl;
						assert(1 == 2);
					}
				}
			} // end if STENCIL TYPE = 0
			/////////////////////////////////////////////////////////////////////////////////////////////
			
			// Create the UMFPACK objects needed to solve Ay = b
			_dim_A = active_points;
			_A_p = new SuiteSparse_long[_dim_A + 1];
			_A_i = new SuiteSparse_long[SuiteSparse_long(val_A.size())];
			_A_x = new double[SuiteSparse_long(val_A.size())];
			_A_z = new double[SuiteSparse_long(val_A.size())];
			_b_x = new double[_dim_A];
			_b_z = new double[_dim_A];
			_y_x = new double[_dim_A];
			_y_z = new double[_dim_A];
			
			this->set_A(_dim_A, _A_p, _A_i, _A_x, _A_z, row_A, col_A, val_A);
			this->set_b(_b_x, _b_z, row_b, val_b);

			double info[UMFPACK_INFO];
			std::cout << "Performing umfpack_zl_symbolic : " << std::endl;
			SuiteSparse_long status_symbolic = umfpack_zl_symbolic(_dim_A, _dim_A, _A_p, _A_i, _A_x, _A_z, &_symbolic, (double*)NULL, info);
			std::cout << "status returned = " << status_symbolic << std::endl;
			
			std::cout << "Performing umfpack_zl_numeric : " << std::endl;
			SuiteSparse_long status_numeric = umfpack_zl_numeric(_A_p, _A_i, _A_x, _A_z, _symbolic, &_numeric, (double*)NULL, info);
			std::cout << "status returned = " << status_numeric << std::endl;

			_is_matrix_ready = true;
			_is_rhs_ready = true;

			return;
		}

		void SparseDirectSolver2D::solve() {
			if (this->is_solver_ready()) {
				double info[UMFPACK_INFO];
				SuiteSparse_long status = umfpack_zl_solve(UMFPACK_A, _A_p, _A_i, _A_x, _A_z, _y_x, _y_z, _b_x, _b_z, _numeric, info, (double*)NULL);
				std::cout << "Performing umfpack_zl_solve : " << std::endl;
				std::cout << "status returned = " << status << std::endl;
			}
			else {
				std::cerr << "Solver not ready. Create matrix and rhs before solving." << std::endl;
				return;
			}
		}

		void SparseDirectSolver2D::extract_solution(Godzilla::Field2D &solution2D) const {
			if (this->is_solver_ready()) {
				if (_forcing2D->get_geom2D().is_equal(solution2D.get_geom2D())) {

					waveX::LockManager<Godzilla::Field2D, Godzilla::vecxd> lock(solution2D, 1);
					Godzilla::xd *ptr_solution;
					const double *ptr_y_x = _y_x;
					const double *ptr_y_z = _y_z;

					const size_t nX_solution = _vel2D->get_geom2D().get_nX();
					const size_t nY_solution = _vel2D->get_geom2D().get_nY();

					const size_t start_nX = (_bc2D->get_bc_face1() != "NBC") ? 1 : 0;
					const size_t end_nX = (_bc2D->get_bc_face3() != "NBC") ? (_sim_geom2D.get_nX() - 2) : (_sim_geom2D.get_nX() - 1);
					const size_t start_nY = (_bc2D->get_bc_face4() != "NBC") ? 1 : 0;
					const size_t end_nY = (_bc2D->get_bc_face2() != "NBC") ? (_sim_geom2D.get_nY() - 2) : (_sim_geom2D.get_nY() - 1);
					const size_t points_X = (start_nX <= end_nX) ? (end_nX - start_nX) + 1 : 0;
					const size_t points_Y = (start_nY <= end_nY) ? (end_nY - start_nY) + 1 : 0;
					const size_t active_points = points_X * points_Y;

					// Fill the boundaries that can be filled directly from boundary conditions
					// If face1 is DBC
					ptr_solution = lock._ptr->data();
					if (_bc2D->get_bc_face1() == "DBC") {
						const Godzilla::xd *ptr_bc_data = _bc2D->get_data1().data();
						for (size_t i = 0; i < nY_solution; ++i) {
							*ptr_solution = ptr_bc_data[i];
							ptr_solution += nX_solution;
						}
					}
					// If face2 is DBC
					ptr_solution = lock._ptr->data();
					ptr_solution += (nY_solution - 1) * nX_solution;
					if (_bc2D->get_bc_face2() == "DBC") {
						const Godzilla::xd *ptr_bc_data = _bc2D->get_data2().data();
						for (size_t i = 0; i < nX_solution; ++i) {
							*ptr_solution = ptr_bc_data[i];
							++ptr_solution;
						}
					}
					// If face3 is DBC
					ptr_solution = lock._ptr->data();
					ptr_solution += nX_solution - 1;
					if (_bc2D->get_bc_face3() == "DBC") {
						const Godzilla::xd *ptr_bc_data = _bc2D->get_data3().data();
						for (size_t i = 0; i < nY_solution; ++i) {
							*ptr_solution = ptr_bc_data[i];
							ptr_solution += nX_solution;
						}
					}
					// If face4 is DBC
					ptr_solution = lock._ptr->data();
					if (_bc2D->get_bc_face4() == "DBC") {
						const Godzilla::xd *ptr_bc_data = _bc2D->get_data4().data();
						for (size_t i = 0; i < nX_solution; ++i) {
							*ptr_solution = ptr_bc_data[i];
							++ptr_solution;
						}
					}
					
					// If active points > 0, can update the points in the intersection
					if (active_points > 0) {
						size_t rel_start_nX = (_bc2D->get_bc_face1() == "PML") ? _pad1 - 1 : 0;
						size_t rel_start_nY = (_bc2D->get_bc_face4() == "PML") ? _pad4 - 1 : 0;
						size_t rel_end_nX = (_bc2D->get_bc_face3() == "PML") ? points_X - _pad3 : points_X - 1;
						size_t rel_end_nY = (_bc2D->get_bc_face2() == "PML") ? points_Y - _pad2 : points_Y - 1;
						size_t points_X_intersection = rel_end_nX - rel_start_nX + 1;
						size_t points_Y_intersection = rel_end_nY - rel_start_nY + 1;

						size_t rel_start_nX_out = (_bc2D->get_bc_face1() == "DBC") ? 1 : 0;
						size_t rel_start_nY_out = (_bc2D->get_bc_face4() == "DBC") ? 1 : 0;

						ptr_solution = lock._ptr->data();
						ptr_solution += rel_start_nY_out * nX_solution + rel_start_nY_out;
						ptr_y_x += rel_start_nY * points_X + rel_start_nX;
						ptr_y_z += rel_start_nY * points_X + rel_start_nX;
						for (size_t i = 0; i < points_Y_intersection; ++i) {
							for (size_t j = 0; j < points_X_intersection; ++j) {
								ptr_solution[j].real(ptr_y_x[j]);
								ptr_solution[j].imag(ptr_y_z[j]);
							}
							ptr_solution += nX_solution;
							ptr_y_x += points_X;
							ptr_y_z += points_X;
						}
					}
					
					lock.deactivate_lock(solution2D);
					return;
				}
				else {
					std::cerr << "Input geometry does not match existing geometry. Cannot extract solution." << std::endl;
					return;
				}
			}
			else {
				std::cerr << "Solver not ready. Create matrix and rhs, and solve before extracting solution." << std::endl;
				return;
			}
		}

		void SparseDirectSolver2D::free_solver_resources() {
			_is_matrix_ready = false;
			_is_rhs_ready = false;
			_dim_A = 0;
			delete[] _A_p;
			delete[] _A_i;
			delete[] _A_x;
			delete[] _A_z;
			delete[] _y_x;
			delete[] _y_z;
			delete[] _b_x;
			delete[] _b_z;
			if (_symbolic != nullptr) umfpack_zl_free_symbolic(&_symbolic);
			if (_numeric != nullptr)umfpack_zl_free_numeric(&_numeric);
			_A_p = nullptr;
			_A_i = nullptr;
			_A_x = nullptr;
			_A_z = nullptr;
			_y_x = nullptr;
			_y_z = nullptr;
			_b_x = nullptr;
			_b_z = nullptr;
			_symbolic = nullptr;
			_numeric = nullptr;
		}

		void SparseDirectSolver2D::print_matrix(const double &umfpack_prl) const {
			if (this->is_solver_ready()) {
				double control[UMFPACK_CONTROL];
				control[UMFPACK_PRL] = umfpack_prl;
				std::cout << "\nPrinting A using umfpack_prl = " << umfpack_prl << " : " << std::endl;
				SuiteSparse_long status = umfpack_zl_report_matrix(_dim_A, _dim_A, _A_p, _A_i, _A_x, _A_z, 1, control);
			}
			else {
				std::cout << "Cannot print A as solver is not ready." << std::endl;
			}
		}

		void SparseDirectSolver2D::print_rhs() const {
			if (this->is_solver_ready()) {
				std::cout << "\nPrinting rhs :" << std::endl;
				for (size_t i = 0; i < _dim_A; ++i) {
					std::cout << "rhs[" << i << "] = " << "( " << _b_x[i] << " , " << _b_z[i] << " )" << std::endl;
				}
			}
			else {
				std::cout << "Cannot print rhs as solver is not ready." << std::endl;
			}
		}

		void SparseDirectSolver2D::print_solution() const {
			if (this->is_solver_ready()) {
				std::cout << "\nPrinting solution :" << std::endl;
				for (size_t i = 0; i < _dim_A; ++i) {
					std::cout << "sol[" << i << "] = " << "( " << _y_x[i] << " , " << _y_z[i] << " )" << std::endl;
				}
			}
			else {
				std::cout << "Cannot print solution as solver is not ready." << std::endl;
			}
		}

		//***********************************************************************************************
		// Private methods
		bool SparseDirectSolver2D::is_velocity_real(const Godzilla::Velocity2D &vel2D_in) const {
			bool flag = true;
			const size_t nelem = vel2D_in.get_nelem();
			const Godzilla::xd *ptr_data = vel2D_in.get_cdata().data();
			for (size_t i = 0; i < nelem; ++i) {
				if (ptr_data[i].imag() != 0) {
					flag = false;
					break;
				}
			}
			return flag;
		}

		bool SparseDirectSolver2D::is_valid_stencil(const int &stencil_type) const {
			if (stencil_type == 0) return true;
			return false;
		}

		Godzilla::Geometry2D SparseDirectSolver2D::create_sim_geom2D(const Godzilla::BoundaryCondition2D &bc2D) const {
			
			// Get geometry
			const Godzilla::Geometry2D &geom2D = bc2D.get_geom2D();

			// Default n1, n2, n3, n4 for NBC, DBC
			size_t n1 = 0;
			size_t n2 = 0;
			size_t n3 = 0;
			size_t n4 = 0;

			if (bc2D.get_bc_face1() == "PML") n1 = bc2D.get_pmlcells_face1();
			if (bc2D.get_bc_face2() == "PML") n2 = bc2D.get_pmlcells_face2();
			if (bc2D.get_bc_face3() == "PML") n3 = bc2D.get_pmlcells_face3();
			if (bc2D.get_bc_face4() == "PML") n4 = bc2D.get_pmlcells_face4();

			const size_t nX = geom2D.get_nX() + n1 + n3;
			const size_t nY = geom2D.get_nY() + n2 + n4;
			const double startX = geom2D.get_startX() - n1 * geom2D.get_hX();
			const double startY = geom2D.get_startY() - n4 * geom2D.get_hY();

			Godzilla::Geometry2D sim_geom2D(geom2D);
			sim_geom2D.set_geometry2D(nX, startX, geom2D.get_hX(), nY, startY, geom2D.get_hY());
			return sim_geom2D;
		}
		
		size_t SparseDirectSolver2D::calculate_pad1(const Godzilla::BoundaryCondition2D &bc2D) const {
			return (bc2D.get_bc_face1() == "PML") ? bc2D.get_pmlcells_face1() : 0;
		}

		size_t SparseDirectSolver2D::calculate_pad2(const Godzilla::BoundaryCondition2D &bc2D) const {
			return (bc2D.get_bc_face2() == "PML") ? bc2D.get_pmlcells_face2() : 0;
		}

		size_t SparseDirectSolver2D::calculate_pad3(const Godzilla::BoundaryCondition2D &bc2D) const {
			return (bc2D.get_bc_face3() == "PML") ? bc2D.get_pmlcells_face3() : 0;
		}

		size_t SparseDirectSolver2D::calculate_pad4(const Godzilla::BoundaryCondition2D &bc2D) const {
			return (bc2D.get_bc_face4() == "PML") ? bc2D.get_pmlcells_face4() : 0;
		}

		Godzilla::vecxd SparseDirectSolver2D::field_simgrid_zeropad(const Godzilla::Field2D &field2D_in, const Godzilla::Geometry2D &geom2D_out,
																	const size_t &pad1, const size_t &pad4) const {

			const size_t nX_out = geom2D_out.get_nX();
			const size_t nY_out = geom2D_out.get_nY();
			Godzilla::vecxd out(nX_out * nY_out, 0.);

			const size_t nX_in = field2D_in.get_geom2D().get_nX();
			const size_t nY_in = field2D_in.get_geom2D().get_nY();

			const Godzilla::xd *ptr_in = field2D_in.get_cdata().data();
			Godzilla::xd *ptr_out = out.data();

			ptr_out += pad4 * nX_out + pad1;
			for (size_t i = 0; i < nY_in; ++i) {
				for (size_t j = 0; j < nX_in; ++j) {
					ptr_out[j] = *ptr_in;
					++ptr_in;
				}
				ptr_out += nX_out;
			}

			return out;
		}
		
		Godzilla::vecxd SparseDirectSolver2D::velocity_simgrid_extrap(const Godzilla::Velocity2D &vel2D_in, const Godzilla::Geometry2D &geom2D_out,
																	  const size_t &pad1, const size_t &pad2, const size_t &pad3, const size_t &pad4) const {

			const size_t nX_out = geom2D_out.get_nX();
			const size_t nY_out = geom2D_out.get_nY();
			Godzilla::vecxd out(nX_out * nY_out, 0.);

			const size_t nX_in = vel2D_in.get_geom2D().get_nX();
			const size_t nY_in = vel2D_in.get_geom2D().get_nY();

			const Godzilla::xd *ptr_in = vel2D_in.get_cdata().data();
			Godzilla::xd *ptr_out = out.data();

			// Copy elements except padding
			ptr_out += pad4 * nX_out + pad1;
			for (size_t i = 0; i < nY_in; ++i) {
				for (size_t j = 0; j < nX_in; ++j) {
					ptr_out[j] = *ptr_in;
					++ptr_in;
				}
				ptr_out += nX_out;
			}

			// Extrapolate left and right padded zones horizontally from edges (except top and bottom zones)
			ptr_out = out.data();
			ptr_out += pad4 * nX_out;
			Godzilla::xd temp;
			for (size_t i = 0; i < nY_in; ++i) {
				temp = ptr_out[pad1];
				for (size_t j = 0; j < pad1; ++j) {
					ptr_out[j] = temp;
				}
				temp = ptr_out[pad1 + nX_in - 1];
				for (size_t j = 0; j < pad3; ++j) {
					ptr_out[j + pad1 + nX_in] = temp;
				}
				ptr_out += nX_out;
			}

			// Extrapolate top and bottom padded zones vertically from edges
			ptr_out = out.data();
			ptr_in = out.data() + pad4 * nX_out;                   // Reuse ptr_in
			for (size_t i = 0; i < pad4; ++i) {
				for (size_t j = 0; j < nX_out; ++j) {
					ptr_out[j] = ptr_in[j];
				}
				ptr_out += nX_out;
			}
			ptr_out = out.data() + nX_out * (pad4 + nY_in);
			ptr_in = out.data() + nX_out * (pad4 + nY_in - 1);     // Reuse ptr_in
			for (size_t i = 0; i < pad2; ++i) {
				for (size_t j = 0; j < nX_out; ++j) {
					ptr_out[j] = ptr_in[j];
				}
				ptr_out += nX_out;
			}

			return out;
		}

		Godzilla::vecxd SparseDirectSolver2D::calculate_sim_bc_face1(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
			                                                         const size_t &pad4, const size_t &pad2) const {
			
			const Godzilla::xd *ptr_in = bc2D.get_data1().data();
			const size_t nY_in = bc2D.get_geom2D().get_nY();

			Godzilla::vecxd out(geom2D_out.get_nY(), 0.);

			if (bc2D.get_bc_face1() != "PML") {
				Godzilla::xd *ptr_out = out.data();
				for (size_t i = 0; i < pad4; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
				for (size_t i = 0; i < nY_in; ++i) {
					*ptr_out = ptr_in[i];
					++ptr_out;
				}
				ptr_in += nY_in - 1;
				for (size_t i = 0; i < pad2; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
			}
			return out;
		}

		Godzilla::vecxd SparseDirectSolver2D::calculate_sim_bc_face2(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
																	 const size_t &pad1, const size_t &pad3) const {

			const Godzilla::xd *ptr_in = bc2D.get_data2().data();
			const size_t nX_in = bc2D.get_geom2D().get_nX();

			Godzilla::vecxd out(geom2D_out.get_nX(), 0.);

			if (bc2D.get_bc_face2() != "PML") {
				Godzilla::xd *ptr_out = out.data();
				for (size_t i = 0; i < pad1; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
				for (size_t i = 0; i < nX_in; ++i) {
					*ptr_out = ptr_in[i];
					++ptr_out;
				}
				ptr_in += nX_in - 1;
				for (size_t i = 0; i < pad3; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
			}
			return out;
		}

		Godzilla::vecxd SparseDirectSolver2D::calculate_sim_bc_face3(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
																	 const size_t &pad4, const size_t &pad2) const {

			const Godzilla::xd *ptr_in = bc2D.get_data3().data();
			const size_t nY_in = bc2D.get_geom2D().get_nY();

			Godzilla::vecxd out(geom2D_out.get_nY(), 0.);

			if (bc2D.get_bc_face3() != "PML") {
				Godzilla::xd *ptr_out = out.data();
				for (size_t i = 0; i < pad4; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
				for (size_t i = 0; i < nY_in; ++i) {
					*ptr_out = ptr_in[i];
					++ptr_out;
				}
				ptr_in += nY_in - 1;
				for (size_t i = 0; i < pad2; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
			}
			return out;
		}

		Godzilla::vecxd SparseDirectSolver2D::calculate_sim_bc_face4(const Godzilla::BoundaryCondition2D &bc2D, const Godzilla::Geometry2D &geom2D_out,
																	 const size_t &pad1, const size_t &pad3) const {

			const Godzilla::xd *ptr_in = bc2D.get_data4().data();
			const size_t nX_in = bc2D.get_geom2D().get_nX();

			Godzilla::vecxd out(geom2D_out.get_nX(), 0.);

			if (bc2D.get_bc_face4() != "PML") {
				Godzilla::xd *ptr_out = out.data();
				for (size_t i = 0; i < pad1; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
				for (size_t i = 0; i < nX_in; ++i) {
					*ptr_out = ptr_in[i];
					++ptr_out;
				}
				ptr_in += nX_in - 1;
				for (size_t i = 0; i < pad3; ++i) {
					*ptr_out = *ptr_in;
					++ptr_out;
				}
			}
			return out;
		}

		Godzilla::vecxd SparseDirectSolver2D::calculate_sX(const Godzilla::vecd &x, const Godzilla::Geometry2D &geom2D, const double &omega,
														   const size_t &pad1, const size_t &pad3) const {
			const double startX = geom2D.get_startX();
			const double endX = geom2D.get_endX();
			const double start_bcX = startX + geom2D.get_hX() * pad1;
			const double end_bcX = endX - geom2D.get_hX() * pad3;
			const double pmlwidth_face1 = std::abs(start_bcX - startX);
			const double pmlwidth_face3 = std::abs(endX - end_bcX);

			const size_t nelem = x.size();
			const double *ptr_in = x.data();
			Godzilla::vecxd out(nelem, 0.);
			Godzilla::xd *ptr_out = out.data();

			const Godzilla::xd j(0., 1.);
			const Godzilla::xd j1(1., 0.);

			if (geom2D.get_hX() > 0) {
				double sigmaX = 0.;
				for (size_t i = 0; i < nelem; ++i) {
					if ((ptr_in[i] >= start_bcX) && (ptr_in[i] <= end_bcX)) {
						sigmaX = 0.;
					}
					else if (ptr_in[i] < start_bcX) {
						sigmaX = (Godzilla::PML_DAMPING / pmlwidth_face1) * std::pow((ptr_in[i] - start_bcX) / pmlwidth_face1, 2.0);
					}
					else if (ptr_in[i] > end_bcX) {
						sigmaX = (Godzilla::PML_DAMPING / pmlwidth_face3) * std::pow((ptr_in[i] - end_bcX) / pmlwidth_face3, 2.0);
					}
					sigmaX /= omega;
					ptr_out[i] = j1 / (j1 + sigmaX * j);
				}
			}
			else {
				double sigmaX = 0.;
				for (size_t i = 0; i < nelem; ++i) {
					if ((ptr_in[i] <= start_bcX) && (ptr_in[i] >= end_bcX)) {
						sigmaX = 0.;
					}
					else if (ptr_in[i] > start_bcX) {
						sigmaX = (Godzilla::PML_DAMPING / pmlwidth_face1) * std::pow((ptr_in[i] - start_bcX) / pmlwidth_face1, 2.0);
					}
					else if (ptr_in[i] < end_bcX) {
						sigmaX = (Godzilla::PML_DAMPING / pmlwidth_face3) * std::pow((ptr_in[i] - end_bcX) / pmlwidth_face3, 2.0);
					}
					sigmaX /= omega;
					ptr_out[i] = j1 / (j1 + sigmaX * j);
				}
			}
			return out;
		}

		Godzilla::vecxd SparseDirectSolver2D::calculate_sY(const Godzilla::vecd &y, const Godzilla::Geometry2D &geom2D, const double &omega,
														   const size_t &pad4, const size_t &pad2) const {
			const double startY = geom2D.get_startY();
			const double endY = geom2D.get_endY();
			const double start_bcY = startY + geom2D.get_hY() * pad4;
			const double end_bcY = endY - geom2D.get_hY() * pad2;
			const double pmlwidth_face4 = start_bcY - startY;
			const double pmlwidth_face2 = endY - end_bcY;

			const size_t nelem = y.size();
			const double *ptr_in = y.data();
			Godzilla::vecxd out(nelem, 0.);
			Godzilla::xd *ptr_out = out.data();

			const Godzilla::xd j(0., 1.);
			const Godzilla::xd j1(1., 0.);

			if (geom2D.get_hY() > 0) {
				double sigmaY = 0.;
				for (size_t i = 0; i < nelem; ++i) {
					if ((ptr_in[i] >= start_bcY) && (ptr_in[i] <= end_bcY)) {
						sigmaY = 0.;
					}
					else if (ptr_in[i] < start_bcY) {
						sigmaY = (Godzilla::PML_DAMPING / pmlwidth_face4) * std::pow((ptr_in[i] - start_bcY) / pmlwidth_face4, 2.0);
					}
					else if (ptr_in[i] > end_bcY) {
						sigmaY = (Godzilla::PML_DAMPING / pmlwidth_face2) * std::pow((ptr_in[i] - end_bcY) / pmlwidth_face2, 2.0);
					}
					sigmaY /= omega;
					ptr_out[i] = j1 / (j1 + sigmaY * j);
				}
			}
			else {
				double sigmaY = 0.;
				for (size_t i = 0; i < nelem; ++i) {
					if ((ptr_in[i] <= start_bcY) && (ptr_in[i] >= end_bcY)) {
						sigmaY = 0.;
					}
					else if (ptr_in[i] > start_bcY) {
						sigmaY = (Godzilla::PML_DAMPING / pmlwidth_face4) * std::pow((ptr_in[i] - start_bcY) / pmlwidth_face4, 2.0);
					}
					else if (ptr_in[i] < end_bcY) {
						sigmaY = (Godzilla::PML_DAMPING / pmlwidth_face2) * std::pow((ptr_in[i] - end_bcY) / pmlwidth_face2, 2.0);
					}
					sigmaY /= omega;
					ptr_out[i] = j1 / (j1 + Godzilla::PML_DAMPING * sigmaY * j);
				}
			}
			return out;
		}

		void SparseDirectSolver2D::append_A(std::vector<size_t> &row_A, std::vector<size_t> &col_A, Godzilla::vecxd &val_A,
											const size_t &row, const size_t &col, const Godzilla::xd &val) const {
			row_A.push_back(row);
			col_A.push_back(col);
			val_A.push_back(val);
		}

		void SparseDirectSolver2D::append_b(std::vector<size_t> &row_b, Godzilla::vecxd &val_b, const size_t &row, const Godzilla::xd &val) const {
			row_b.push_back(row);
			val_b.push_back(val);
		}
		
		void SparseDirectSolver2D::set_A(const SuiteSparse_long &n, SuiteSparse_long *A_p, SuiteSparse_long *A_i, double *A_x, double *A_z,
										 const std::vector<size_t> &row_A, const std::vector<size_t> &col_A, const Godzilla::vecxd &val_A) {
			
			const size_t nelem = val_A.size();
			const size_t *ptr_row_A = row_A.data();
			const size_t *ptr_col_A = col_A.data();
			const Godzilla::xd *ptr_val_A = val_A.data();

			/////////////////////////////////////////////////////////////////////////////////////////////
			// Set A_p

			// Set all values to zero
			for (size_t i = 0; i <= n; ++i) {
				A_p[i] = 0;
			}
			// Count number of elements in col i and write the value in A_p[i + 1]
			++A_p;
			for (size_t i = 0; i < nelem; ++i) {
				A_p[ptr_col_A[i]] += 1;
			}
			// Perform a running sum
			for (size_t i = 0; i < n; ++i) {
				A_p[i] += A_p[i - 1];
			}
			--A_p;
			/////////////////////////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////////////////////////
			// Set A_i, A_x, A_z without sorting

			// Make a copy of A_p in temp
			SuiteSparse_long *temp = new SuiteSparse_long[n + 1];
			for (size_t i = 0; i <= n; ++i) {
				temp[i] = A_p[i];
			}
			// Populate A_i, A_x, A_z with columns sorted, but each row inside a column unsorted
			size_t pos;
			for (size_t i = 0; i < nelem; ++i) {
				pos = temp[ptr_col_A[i]];
				A_i[pos] = ptr_row_A[i];
				A_x[pos] = ptr_val_A[i].real();
				A_z[pos] = ptr_val_A[i].imag();
				temp[ptr_col_A[i]] += 1;
			}
			delete[] temp;
			/////////////////////////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////////////////////////
			// Sort A_i, A_x, A_z

			// Define temporary variables
			size_t nelem_col = 0;
			pos = 0;
			// Define new temp arrays and get pointers to A_i, A_x, A_z
			temp = new SuiteSparse_long[nelem];
			SuiteSparse_long *temp1 = new SuiteSparse_long[nelem];
			double *temp2 = new double[nelem];
			SuiteSparse_long *temp_A_i = A_i;
			double *temp_A_x = A_x;
			double *temp_A_z = A_z;
			// Loop over each column and sort the entries
			for (size_t i = 0; i < n; ++i) {
				// Calculate number of non-zero entries in column
				nelem_col = A_p[i + 1] - A_p[i];
				// In temp write elements 0, 1, ... , nelem_col - 1 at positions temp[pos] to temp[pos + nelem_col - 1]
				for (size_t j = 0; j < nelem_col; ++j) temp[pos + j] = j;
				// Sort the elements from temp[pos] to temp[pos + nelem_col - 1]
				std::sort(temp + pos, temp + pos + nelem_col, [&](size_t a, size_t b) { return temp_A_i[a] < temp_A_i[b]; });
				// Copy elements of A_i into temp1 in sorted order and then write back
				for (size_t j = 0; j < nelem_col; ++j) {
					temp1[pos + j] = temp_A_i[temp[pos + j]];
				}
				for (size_t j = 0; j < nelem_col; ++j) {
					temp_A_i[j] = temp1[pos + j];
				}
				// Copy elements of A_x into temp2 in sorted order and then write back
				for (size_t j = 0; j < nelem_col; ++j) {
					temp2[pos + j] = temp_A_x[temp[pos + j]];
				}
				for (size_t j = 0; j < nelem_col; ++j) {
					temp_A_x[j] = temp2[pos + j];
				}
				// Copy elements of A_z into temp2 in sorted order and then write back
				for (size_t j = 0; j < nelem_col; ++j) {
					temp2[pos + j] = temp_A_z[temp[pos + j]];
				}
				for (size_t j = 0; j < nelem_col; ++j) {
					temp_A_z[j] = temp2[pos + j];
				}
				// Adjust all pointers
				pos += nelem_col;
				temp_A_i += nelem_col;
				temp_A_x += nelem_col;
				temp_A_z += nelem_col;
			}
			delete[] temp;
			delete[] temp1;
			delete[] temp2;
			/////////////////////////////////////////////////////////////////////////////////////////////

			return;
		}

		void SparseDirectSolver2D::set_b(double *b_x, double *b_z, const std::vector<size_t> &row_b, const Godzilla::vecxd &val_b) {
			const size_t n = row_b.size();
			const size_t *ptr_row_b = row_b.data();
			const Godzilla::xd *ptr_val_b = val_b.data();
			for (size_t i = 0; i < n; ++i) {
				b_x[ptr_row_b[i]] = ptr_val_b[i].real();
				b_z[ptr_row_b[i]] = ptr_val_b[i].imag();
			}
		}

	}
}