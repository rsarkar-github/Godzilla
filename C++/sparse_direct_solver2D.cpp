#include "sparse_direct_solver2D.h"
#include <cmath>

namespace Godzilla {
	namespace Helmholtz2DReal {
		// Constructors
		SparseDirectSolver2D::SparseDirectSolver2D(const Godzilla::Velocity2D &vel2D, const Godzilla::Field2D &forcing2D,
												   const Godzilla::BoundaryCondition2D &bc2D, const double &omega)
			: _vel2D(nullptr), _forcing2D(nullptr), _bc2D(nullptr), _omega(omega), _initialized_state(false), _is_matrix_ready(false), _is_rhs_ready(false) {

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
				_is_rhs_ready = false;
			}
			else {
				std::cerr << "Input geometry does not match existing geometry. Cannot change velocity data." << std::endl;
			}
		}

		void SparseDirectSolver2D::change_forcing_data(const Godzilla::Field2D &forcing2D) {
			if (_vel2D->get_geom2D().is_equal(forcing2D.get_geom2D())) {
				_forcing2D = &forcing2D;
				_forcing_data_simgrid = this->field_simgrid_zeropad(*_forcing2D, _sim_geom2D, _pad1, _pad4);
				_is_matrix_ready = false;
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
			_omega = omega;
			_is_matrix_ready = false;
		}

		void SparseDirectSolver2D::create_sparse_matrix() {

		}

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

	}
}