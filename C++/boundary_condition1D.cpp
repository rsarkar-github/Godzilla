#include "boundary_condition1D.h"
#include <iostream>

namespace Godzilla {
	// Constructors
	BoundaryCondition1D::BoundaryCondition1D() : _bc_face1(Godzilla::BC_DEFAULT), _bc_face2(Godzilla::BC_DEFAULT),
												 _pmlcells_face1(Godzilla::PML_CELLS_DEFAULT), _pmlcells_face2(Godzilla::PML_CELLS_DEFAULT) {

		_geom1D = Godzilla::Geometry1D();
		_data1 = 0.;
		_data2 = 0.;
	}

	BoundaryCondition1D::BoundaryCondition1D(const Godzilla::Geometry1D &geom1D) : _geom1D(geom1D), _bc_face1(Godzilla::BC_DEFAULT), _bc_face2(Godzilla::BC_DEFAULT),
																				   _pmlcells_face1(Godzilla::PML_CELLS_DEFAULT), _pmlcells_face2(Godzilla::PML_CELLS_DEFAULT) {

		_data1 = 0.;
		_data2 = 0.;
	}

	BoundaryCondition1D::BoundaryCondition1D(const Godzilla::Geometry1D &geom1D, const std::string &bc_face1, const std::string &bc_face2) : _geom1D(geom1D) {
		_bc_face1 = this->is_valid_boundary_condition(bc_face1) ? bc_face1 : Godzilla::BC_DEFAULT;
		_bc_face2 = this->is_valid_boundary_condition(bc_face2) ? bc_face2 : Godzilla::BC_DEFAULT;
		_pmlcells_face1 = Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face2 = Godzilla::PML_CELLS_DEFAULT;

		_data1 = 0.;
		_data2 = 0.;
	}

	BoundaryCondition1D::BoundaryCondition1D(const Godzilla::Geometry1D &geom1D, const std::string &bc_face1,
											 const size_t &pmlcells_face1, const std::string &bc_face2, const size_t &pmlcells_face2) : _geom1D(geom1D){
		
		_bc_face1 = this->is_valid_boundary_condition(bc_face1) ? bc_face1 : Godzilla::BC_DEFAULT;
		_bc_face2 = this->is_valid_boundary_condition(bc_face2) ? bc_face2 : Godzilla::BC_DEFAULT;
		_pmlcells_face1 = ((bc_face1 == "PML") && (pmlcells_face1 >= Godzilla::PML_CELLS_DEFAULT)) ? pmlcells_face1 : Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face2 = ((bc_face2 == "PML") && (pmlcells_face2 >= Godzilla::PML_CELLS_DEFAULT)) ? pmlcells_face2 : Godzilla::PML_CELLS_DEFAULT;
		
		_data1 = 0.;
		_data2 = 0.;
	}

	BoundaryCondition1D::BoundaryCondition1D(const Godzilla::BoundaryCondition1D &bc1D) {
		_bc_face1 = bc1D.get_bc_face1();
		_bc_face2 = bc1D.get_bc_face2();
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
		_geom1D = bc1D.get_geom1D();
		_data1 = bc1D.get_data1();
		_data2 = bc1D.get_data2();
	}

	BoundaryCondition1D::BoundaryCondition1D(Godzilla::BoundaryCondition1D &&bc1D) {
		_bc_face1 = std::move(bc1D._bc_face1);
		_bc_face2 = std::move(bc1D._bc_face2);
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
		_geom1D = std::move(bc1D._geom1D);
		_data1 = std::move(bc1D._data1);
		_data2 = std::move(bc1D._data2);
	}

	// Public methods
	Godzilla::BoundaryCondition1D& BoundaryCondition1D::operator=(const Godzilla::BoundaryCondition1D &bc1D) {
		_bc_face1 = bc1D.get_bc_face1();
		_bc_face2 = bc1D.get_bc_face2();
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
		_geom1D = bc1D.get_geom1D();
		_data1 = bc1D.get_data1();
		_data2 = bc1D.get_data2();
		return *this;
	}

	Godzilla::BoundaryCondition1D& BoundaryCondition1D::operator=(Godzilla::BoundaryCondition1D &&bc1D) {
		_bc_face1 = std::move(bc1D._bc_face1);
		_bc_face2 = std::move(bc1D._bc_face2);
		_pmlcells_face1 = bc1D.get_pmlcells_face1();
		_pmlcells_face2 = bc1D.get_pmlcells_face2();
		_geom1D = std::move(bc1D._geom1D);
		_data1 = std::move(bc1D._data1);
		_data2 = std::move(bc1D._data2);
		return *this;
	}

	void BoundaryCondition1D::set_bc_face1(const std::string &bc_face1) {
		if (this->is_valid_boundary_condition(bc_face1)) {
			_bc_face1 = bc_face1;
			_data1 = (bc_face1 == "PML") ? 0. : _data1;
		}
		else {
			std::cerr << "Not a valid boundary condition. Boundary condition not changed." << std::endl;
		}
	}

	void BoundaryCondition1D::set_bc_face2(const std::string &bc_face2) {
		if (this->is_valid_boundary_condition(bc_face2)) {
			_bc_face2 = bc_face2;
			_data2 = (bc_face2 == "PML") ? 0. : _data2;
		}
		else {
			std::cerr << "Not a valid boundary condition. Boundary condition not changed." << std::endl;
		}
	}

	void BoundaryCondition1D::set_pmlcells_face1(const size_t &pmlcells_face1) {
		if (pmlcells_face1 >= Godzilla::PML_CELLS_DEFAULT) {
			_pmlcells_face1 = (_bc_face1 == "PML") ? pmlcells_face1 : _pmlcells_face1;
		}
		else {
			std::cerr << "Number of PML cells cannot be less than default. PML cells unchanged." << std::endl;
		}
	}

	void BoundaryCondition1D::set_pmlcells_face2(const size_t &pmlcells_face2) {
		if (pmlcells_face2 >= Godzilla::PML_CELLS_DEFAULT) {
			_pmlcells_face2 = (_bc_face2 == "PML") ? pmlcells_face2 : _pmlcells_face2;
		}
		else {
			std::cerr << "Number of PML cells cannot be less than default. PML cells unchanged." << std::endl;
		}
	}
	
	void BoundaryCondition1D::set_data(const Godzilla::xd &data1, const Godzilla::xd &data2) {
		_data1 = (!this->is_PML(_bc_face1)) ? data1 : 0.;
		_data2 = (!this->is_PML(_bc_face2)) ? data2 : 0.;
	}

	// Private methods
	bool BoundaryCondition1D::is_valid_boundary_condition(const std::string &bcface) const {
		if (bcface == "DBC") return true;
		if (bcface == "NBC") return true;
		if (bcface == "PML") return true;
		return false;
	}
}