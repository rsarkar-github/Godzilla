#include "sparse_direct_solver2D.h"

namespace Godzilla {
	namespace Helmholtz2DReal {
		// Constructors
		SparseDirectSolver2D::SparseDirectSolver2D() {

		}

		// Public methods

		// Private methods
		Godzilla::Geometry2D SparseDirectSolver2D::create_sim_geom2D(const Godzilla::BoundaryCondition2D &bc2D) const {
			
			// Get geometry
			Godzilla::Geometry2D &geom2D = bc2D.get_geom2D();

			// Default n1, n2, n3, n4 for NBC, DBC
			size_t n1 = 0;
			size_t n2 = 0;
			size_t n3 = 0;
			size_t n4 = 0;

			if (bc2D.get_bc_face1() == "PML") n1 = bc2D.get_pmlcells_face1();
			if (bc2D.get_bc_face2() == "PML") n2 = bc2D.get_pmlcells_face2();
			if (bc2D.get_bc_face3() == "PML") n3 = bc2D.get_pmlcells_face3();
			if (bc2D.get_bc_face4() == "PML") n4 = bc2D.get_pmlcells_face4();

			size_t nX = geom2D.get_nX() + n1 + n3;
			size_t nY = geom2D.get_nY() + n2 + n4;
			double startX = geom2D.get_startX() - n1 * geom2D.get_hX();
			double startY = geom2D.get_startY() - n4 * geom2D.get_hY();

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
																	const size_t &pad1, const size_t &pad4) {

			size_t nX_out = geom2D_out.get_nX();
			size_t nY_out = geom2D_out.get_nY();
			Godzilla::vecxd out(nX_out * nY_out, 0.);

			size_t nX_in = field2D_in.get_geom2D().get_nX();
			size_t nY_in = field2D_in.get_geom2D().get_nY();

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
																	  const size_t &pad1, const size_t &pad2, const size_t &pad3, const size_t &pad4) {

			size_t nX_out = geom2D_out.get_nX();
			size_t nY_out = geom2D_out.get_nY();
			Godzilla::vecxd out(nX_out * nY_out, 0.);

			size_t nX_in = vel2D_in.get_geom2D().get_nX();
			size_t nY_in = vel2D_in.get_geom2D().get_nY();

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
	}
}