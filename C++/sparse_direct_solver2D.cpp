#include "sparse_direct_solver2D.h"
#include <cmath>

namespace Godzilla {
	namespace Helmholtz2DReal {
		// Constructors
		SparseDirectSolver2D::SparseDirectSolver2D(const Godzilla::Velocity2D &vel2D, const Godzilla::Field2D &forcing2D,
												   const Godzilla::BoundaryCondition2D &bc2D, const double &omega, const int &stencil_type)
			: _vel2D(nullptr), _forcing2D(nullptr), _bc2D(nullptr), _initialized_state(false), _is_matrix_ready(false), _is_rhs_ready(false) {

			_omega = (omega != 0.) ? omega : 1.;
			_stencil_type = (this->is_valid_stencil(stencil_type)) ? stencil_type : 0;

			const Godzilla::Geometry2D &geom2D = vel2D.get_geom2D();
			if (this->is_velocity_real(vel2D) && geom2D.is_equal(forcing2D.get_geom2D()) && geom2D.is_equal(bc2D.get_geom2D())) {
				_vel2D = &vel2D;
				_forcing2D = &forcing2D;
				_bc2D = &bc2D;
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
		void SparseDirectSolver2D::change_velocity_data(const Godzilla::Velocity2D &vel2D) {
			if (_vel2D->get_geom2D().is_equal(vel2D.get_geom2D()) && this->is_velocity_real(vel2D)) {
				_vel2D = &vel2D;
				_velocity_data_simgrid = this->velocity_simgrid_extrap(*_vel2D, _sim_geom2D, _pad1, _pad2, _pad3, _pad4);
				_is_matrix_ready = false;
			}
			else {
				std::cerr << "Input geometry does not match existing geometry. Cannot change velocity data." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_forcing_data(const Godzilla::Field2D &forcing2D) {
			if (_vel2D->get_geom2D().is_equal(forcing2D.get_geom2D())) {
				_forcing2D = &forcing2D;
				_forcing_data_simgrid = this->field_simgrid_zeropad(*_forcing2D, _sim_geom2D, _pad1, _pad4);
				_is_rhs_ready = false;
			}
			else {
				std::cerr << "Input geometry does not match existing geometry. Cannot change forcing data." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_boundary_conditions(const Godzilla::BoundaryCondition2D &bc2D) {
			if (_vel2D->get_geom2D().is_equal(bc2D.get_geom2D())) {
				_bc2D = &bc2D;
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
			}
			else {
				std::cerr << "Input geometry does not match existing geometry. Cannot change boundary conditions." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_omega(const double &omega) {
			if (omega != 0.) {
				_omega = omega;
				_is_matrix_ready = false;
			}
			else {
				_omega = 1.;
			}
		}

		void SparseDirectSolver2D::change_stencil_type(const int &stencil_type) {
			if ((stencil_type != _stencil_type) && (this->is_valid_stencil(stencil_type))) {
				_stencil_type = stencil_type;
				_is_matrix_ready = false;
				_is_rhs_ready = false;
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

			const size_t active_points = ((points_X != 0) && (points_Y != 0)) ? points_X * points_Y : 0;
			if (active_points == 0) {
				std::cerr << "No active points to solve for. Boundary conditions determine solution." << std::endl;
				return;
			}

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
				const Godzilla::vecxd sX = this->calculate_sX(x, _sim_geom2D, _omega, _pad1, _pad3);
				const Godzilla::vecxd sY = this->calculate_sY(y, _sim_geom2D, _omega, _pad4, _pad2);
				
				// Get pointers to the sX, sY, velocity, forcing and boundary conditions along the 4 faces
				const Godzilla::xd *ptr_sX = sX.data();
				const Godzilla::xd *ptr_sY = sY.data();
				const Godzilla::xd *ptr_velocity2D = _velocity_data_simgrid.data();
				const Godzilla::xd *ptr_forcing2D = _forcing_data_simgrid.data();
				const Godzilla::xd *ptr_bc_face1 = _sim_bc_face1.data();
				const Godzilla::xd *ptr_bc_face2 = _sim_bc_face2.data();
				const Godzilla::xd *ptr_bc_face3 = _sim_bc_face3.data();
				const Godzilla::xd *ptr_bc_face4 = _sim_bc_face4.data();

				// Define vectors to create sparse matrix in triplet form
				std::vector<size_t> row_A, col_A, row_b;
				Godzilla::vecxd val_A, val_b;

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

					sY_index = 2 * (start_nY + 1);
					sX_index = 2 * (start_nX + 1);

					p1Y = ptr_sY[sY_index] / hY;
					p2Y = ptr_sY[sY_index + 1] / hY;
					p3Y = ptr_sY[sY_index - 1] / hY;
					
					for (size_t j = 1; j < points_X - 1; ++j) {

						p1X = ptr_sX[sX_index] / hX;
						p2X = ptr_sX[sX_index + 1] / hX;
						p3X = ptr_sX[sX_index - 1] / hX;

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

				p1X = ptr_sX[sX_index] / hX;
				p2X = ptr_sX[sX_index + 1] / hX;
				p3X = ptr_sX[sX_index - 1] / hX;

				for (size_t i = 1; i < points_Y - 1; ++i) {
					p1Y = ptr_sY[sY_index] / hY;
					p2Y = ptr_sY[sY_index + 1] / hY;
					p3Y = ptr_sY[sY_index - 1] / hY;

					// Forcing term
					if (_bc2D->get_bc_face1() != "NBC") {
						val = ptr_forcing2D[forcing_index] - p1X * p3X * ptr_bc_face1[bc_face1_index];
					}
					else {
						val = ptr_forcing2D[forcing_index] - 2 * hX * p1X * p3X * ptr_bc_face1[bc_face1_index];
					}
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Right coefficient
					if (_bc2D->get_bc_face1() != "NBC") {
						val = p1X * p2X;
					}
					else {
						val = p1X * (p2X + p3X);
					}
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

				p1Y = ptr_sY[sY_index] / hY;
				p2Y = ptr_sY[sY_index + 1] / hY;
				p3Y = ptr_sY[sY_index - 1] / hY;

				for (size_t i = 1; i < points_X - 1; ++i) {
						
					p1X = ptr_sX[sX_index] / hX;
					p2X = ptr_sX[sX_index + 1] / hX;
					p3X = ptr_sX[sX_index - 1] / hX;

					// Forcing term
					if (_bc2D->get_bc_face2() != "NBC") {
						val = ptr_forcing2D[forcing_index] - p1Y * p2Y * ptr_bc_face2[bc_face2_index];
					}
					else {
						val = ptr_forcing2D[forcing_index] - 2 * hY * p1Y * p2Y * ptr_bc_face2[bc_face2_index];
					}
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
					if (_bc2D->get_bc_face2() != "NBC") {
						val = p1Y * p3Y;
					}
					else {
						val = p1Y * (p2Y + p3Y);
					}
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

				p1X = ptr_sX[sX_index] / hX;
				p2X = ptr_sX[sX_index + 1] / hX;
				p3X = ptr_sX[sX_index - 1] / hX;

				for (size_t i = 1; i < points_Y - 1; ++i) {

					p1Y = ptr_sY[sY_index] / hY;
					p2Y = ptr_sY[sY_index + 1] / hY;
					p3Y = ptr_sY[sY_index - 1] / hY;

					// Forcing term
					row_b.push_back(n_index);
					if (_bc2D->get_bc_face3() != "NBC") {
						val = ptr_forcing2D[forcing_index] - p1X * p2X * ptr_bc_face3[bc_face3_index];
					}
					else {
						val = ptr_forcing2D[forcing_index] - 2 * hX * p1X * p2X * ptr_bc_face3[bc_face3_index];
					}
					this->append_b(row_b, val_b, n_index, val);

					// Central coefficient
					val = -p1X * (p2X + p3X) - p1Y * (p2Y + p3Y) + std::pow(f / ptr_velocity2D[velocity_index], 2.);
					this->append_A(row_A, col_A, val_A, n_index, n_index, val);

					// Left coefficient
					if (_bc2D->get_bc_face3() != "NBC") {
						val = p1X * p3X;
					}
					else {
						val = p1X * (p2X + p3X);
					}
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

				p1Y = ptr_sY[sY_index] / hY;
				p2Y = ptr_sY[sY_index + 1] / hY;
				p3Y = ptr_sY[sY_index - 1] / hY;

				for (size_t i = 1; i < points_X - 1; ++i) {

					p1X = ptr_sX[sX_index] / hX;
					p2X = ptr_sX[sX_index + 1] / hX;
					p3X = ptr_sX[sX_index - 1] / hX;

					// Forcing term
					row_b.push_back(n_index);
					if (_bc2D->get_bc_face4() != "NBC") {
						val = ptr_forcing2D[forcing_index] - p1Y * p2Y * ptr_bc_face4[bc_face4_index];
					}
					else {
						val = ptr_forcing2D[forcing_index] - 2 * hY * p1Y * p3Y * ptr_bc_face4[bc_face4_index];
					}
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
					if (_bc2D->get_bc_face4() != "NBC") {
						val = p1Y * p2Y;
					}
					else {
						val = p1Y * (p2Y + p3Y);
					}
					this->append_A(row_A, col_A, val_A, n_index, n_index + points_X, val);

					sX_index += 2;
					++n_index;
					++velocity_index;
					++forcing_index;
					++bc_face4_index;
				}

				// Handle face1-face2 corners (top left)

			} // end if

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
			if (bc2D.get_bc_face1() == "PML") {
				return bc2D.get_pmlcells_face1();
			}
			else {
				return 0;
			}
		}

		size_t SparseDirectSolver2D::calculate_pad2(const Godzilla::BoundaryCondition2D &bc2D) const {
			if (bc2D.get_bc_face2() == "PML") {
				return bc2D.get_pmlcells_face2();
			}
			else {
				return 0;
			}
		}

		size_t SparseDirectSolver2D::calculate_pad3(const Godzilla::BoundaryCondition2D &bc2D) const {
			if (bc2D.get_bc_face3() == "PML") {
				return bc2D.get_pmlcells_face3();
			}
			else {
				return 0;
			}
		}

		size_t SparseDirectSolver2D::calculate_pad4(const Godzilla::BoundaryCondition2D &bc2D) const {
			if (bc2D.get_bc_face4() == "PML") {
				return bc2D.get_pmlcells_face4();
			}
			else {
				return 0;
			}
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
			const double pmlwidth_face1 = start_bcX - startX;
			const double pmlwidth_face3 = endX - end_bcX;

			const size_t nelem = x.size();
			const double *ptr_in = x.data();
			Godzilla::vecxd out(nelem, 0.);
			Godzilla::xd * ptr_out = out.data();

			const Godzilla::xd j(0., 1.);
			const Godzilla::xd j1(1., 0.);

			if (geom2D.get_hX() > 0) {
				double sigmaX = 0.;
				for (size_t i = 0; i < nelem; ++i) {
					if ((ptr_in[i] >= start_bcX) && (ptr_in[i] <= end_bcX)) {
						sigmaX = 0.;
					}
					else if (ptr_in[i] < start_bcX) {
						sigmaX = std::pow((ptr_in[i] - start_bcX) / pmlwidth_face1, 2.0);
					}
					else if (ptr_in[i] > end_bcX) {
						sigmaX = std::pow((ptr_in[i] - end_bcX) / pmlwidth_face3, 2.0);
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
						sigmaX = std::pow((ptr_in[i] - start_bcX) / pmlwidth_face1, 2.0);
					}
					else if (ptr_in[i] < end_bcX) {
						sigmaX = std::pow((ptr_in[i] - end_bcX) / pmlwidth_face3, 2.0);
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
			Godzilla::xd * ptr_out = out.data();

			const Godzilla::xd j(0., 1.);
			const Godzilla::xd j1(1., 0.);

			if (geom2D.get_hY() > 0) {
				double sigmaY = 0.;
				for (size_t i = 0; i < nelem; ++i) {
					if ((ptr_in[i] >= start_bcY) && (ptr_in[i] <= end_bcY)) {
						sigmaY = 0.;
					}
					else if (ptr_in[i] < start_bcY) {
						sigmaY = std::pow((ptr_in[i] - start_bcY) / pmlwidth_face4, 2.0);
					}
					else if (ptr_in[i] > end_bcY) {
						sigmaY = std::pow((ptr_in[i] - end_bcY) / pmlwidth_face2, 2.0);
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
						sigmaY = std::pow((ptr_in[i] - start_bcY) / pmlwidth_face4, 2.0);
					}
					else if (ptr_in[i] < end_bcY) {
						sigmaY = std::pow((ptr_in[i] - end_bcY) / pmlwidth_face2, 2.0);
					}
					sigmaY /= omega;
					ptr_out[i] = j1 / (j1 + sigmaY * j);
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
	}
}