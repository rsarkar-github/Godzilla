#include "boundary_condition2D.h"
#include <iostream>

namespace Godzilla {
	// Constructors
	BoundaryCondition2D::BoundaryCondition2D() : _bc_face1(Godzilla::BC_DEFAULT), _bc_face2(Godzilla::BC_DEFAULT),
												 _bc_face3(Godzilla::BC_DEFAULT), _bc_face4(Godzilla::BC_DEFAULT),
												 _pmlcells_face1(Godzilla::PML_CELLS_DEFAULT), _pmlcells_face2(Godzilla::PML_CELLS_DEFAULT),
												 _pmlcells_face3(Godzilla::PML_CELLS_DEFAULT), _pmlcells_face4(Godzilla::PML_CELLS_DEFAULT) {
		
		_geom2D = Godzilla::Geometry2D();
		_data1.assign(_geom2D.get_nY(), 0.);
		_data1.shrink_to_fit();
		_data2.assign(_geom2D.get_nX(), 0.);
		_data2.shrink_to_fit();
		_data3.assign(_geom2D.get_nY(), 0.);
		_data3.shrink_to_fit();
		_data4.assign(_geom2D.get_nX(), 0.);
		_data4.shrink_to_fit();
	}

	BoundaryCondition2D::BoundaryCondition2D(const Godzilla::Geometry2D &geom2D, const std::string &bc_face1, const std::string &bc_face2,
											 const std::string &bc_face3, const std::string &bc_face4) : _geom2D(geom2D) {
		
		_bc_face1 = this->is_valid_boundary_condition(bc_face1) ? bc_face1 : Godzilla::BC_DEFAULT;
		_bc_face2 = this->is_valid_boundary_condition(bc_face2) ? bc_face2 : Godzilla::BC_DEFAULT;
		_bc_face3 = this->is_valid_boundary_condition(bc_face3) ? bc_face3 : Godzilla::BC_DEFAULT;
		_bc_face4 = this->is_valid_boundary_condition(bc_face4) ? bc_face4 : Godzilla::BC_DEFAULT;
		_pmlcells_face1 = Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face2 = Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face3 = Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face4 = Godzilla::PML_CELLS_DEFAULT;

		if (!this->is_PML(_bc_face1)) {
			_data1.assign(_geom2D.get_nY(), 0.);
			_data1.shrink_to_fit();
		}
		if (!this->is_PML(_bc_face2)) {
			_data2.assign(_geom2D.get_nX(), 0.);
			_data2.shrink_to_fit();
		}
		if (!this->is_PML(_bc_face3)) {
			_data3.assign(_geom2D.get_nY(), 0.);
			_data3.shrink_to_fit();
		}
		if (!this->is_PML(_bc_face4)) {
			_data4.assign(_geom2D.get_nX(), 0.);
			_data4.shrink_to_fit();
		}
	}

	BoundaryCondition2D::BoundaryCondition2D(const Godzilla::Geometry2D &geom2D, const std::string &bc_face1, const size_t &pmlcells_face1,
											 const std::string &bc_face2, const size_t &pmlcells_face2,
											 const std::string &bc_face3, const size_t &pmlcells_face3,
											 const std::string &bc_face4, const size_t &pmlcells_face4) : _geom2D(geom2D) {
		
		_bc_face1 = this->is_valid_boundary_condition(bc_face1) ? bc_face1 : Godzilla::BC_DEFAULT;
		_bc_face2 = this->is_valid_boundary_condition(bc_face2) ? bc_face2 : Godzilla::BC_DEFAULT;
		_bc_face3 = this->is_valid_boundary_condition(bc_face3) ? bc_face3 : Godzilla::BC_DEFAULT;
		_bc_face4 = this->is_valid_boundary_condition(bc_face4) ? bc_face4 : Godzilla::BC_DEFAULT;

		_pmlcells_face1 = ((bc_face1 == "PML") && (pmlcells_face1 >= Godzilla::PML_CELLS_DEFAULT)) ? pmlcells_face1 : Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face2 = ((bc_face2 == "PML") && (pmlcells_face2 >= Godzilla::PML_CELLS_DEFAULT)) ? pmlcells_face2 : Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face3 = ((bc_face3 == "PML") && (pmlcells_face3 >= Godzilla::PML_CELLS_DEFAULT)) ? pmlcells_face3 : Godzilla::PML_CELLS_DEFAULT;
		_pmlcells_face4 = ((bc_face4 == "PML") && (pmlcells_face4 >= Godzilla::PML_CELLS_DEFAULT)) ? pmlcells_face4 : Godzilla::PML_CELLS_DEFAULT;

		if (!this->is_PML(_bc_face1)) {
			_data1.assign(_geom2D.get_nY(), 0.);
			_data1.shrink_to_fit();
		}
		if (!this->is_PML(_bc_face2)) {
			_data2.assign(_geom2D.get_nX(), 0.);
			_data2.shrink_to_fit();
		}
		if (!this->is_PML(_bc_face3)) {
			_data3.assign(_geom2D.get_nY(), 0.);
			_data3.shrink_to_fit();
		}
		if (!this->is_PML(_bc_face4)) {
			_data4.assign(_geom2D.get_nX(), 0.);
			_data4.shrink_to_fit();
		}
	}

	BoundaryCondition2D::BoundaryCondition2D(const Godzilla::BoundaryCondition2D &bc2D) {
		_bc_face1 = bc2D.get_bc_face1();
		_bc_face2 = bc2D.get_bc_face2();
		_bc_face3 = bc2D.get_bc_face3();
		_bc_face4 = bc2D.get_bc_face4();
		_pmlcells_face1 = bc2D.get_pmlcells_face1();
		_pmlcells_face2 = bc2D.get_pmlcells_face2();
		_pmlcells_face3 = bc2D.get_pmlcells_face3();
		_pmlcells_face4 = bc2D.get_pmlcells_face4();
		_geom2D = bc2D.get_geom2D();
		_data1 = bc2D.get_data1();
		_data2 = bc2D.get_data2();
		_data3 = bc2D.get_data3();
		_data4 = bc2D.get_data4();
	}

	BoundaryCondition2D::BoundaryCondition2D(Godzilla::BoundaryCondition2D &&bc2D) {
		_bc_face1 = std::move(bc2D._bc_face1);
		_bc_face2 = std::move(bc2D._bc_face2);
		_bc_face3 = std::move(bc2D._bc_face3);
		_bc_face4 = std::move(bc2D._bc_face4);
		_pmlcells_face1 = bc2D.get_pmlcells_face1();
		_pmlcells_face2 = bc2D.get_pmlcells_face2();
		_pmlcells_face3 = bc2D.get_pmlcells_face3();
		_pmlcells_face4 = bc2D.get_pmlcells_face4();
		_geom2D = std::move(bc2D._geom2D);
		_data1 = std::move(bc2D._data1);
		_data2 = std::move(bc2D._data2);
		_data3 = std::move(bc2D._data3);
		_data4 = std::move(bc2D._data4);
	}

	// Public methods
	Godzilla::BoundaryCondition2D& BoundaryCondition2D::operator=(const Godzilla::BoundaryCondition2D &bc2D) {
		_bc_face1 = bc2D.get_bc_face1();
		_bc_face2 = bc2D.get_bc_face2();
		_bc_face3 = bc2D.get_bc_face3();
		_bc_face4 = bc2D.get_bc_face4();
		_pmlcells_face1 = bc2D.get_pmlcells_face1();
		_pmlcells_face2 = bc2D.get_pmlcells_face2();
		_pmlcells_face3 = bc2D.get_pmlcells_face3();
		_pmlcells_face4 = bc2D.get_pmlcells_face4();
		_geom2D = bc2D.get_geom2D();
		_data1 = bc2D.get_data1();
		_data2 = bc2D.get_data2();
		_data3 = bc2D.get_data3();
		_data4 = bc2D.get_data4();
		return *this;
	}

	Godzilla::BoundaryCondition2D& BoundaryCondition2D::operator=(Godzilla::BoundaryCondition2D &&bc2D) {
		_bc_face1 = std::move(bc2D._bc_face1);
		_bc_face2 = std::move(bc2D._bc_face2);
		_bc_face3 = std::move(bc2D._bc_face3);
		_bc_face4 = std::move(bc2D._bc_face4);
		_pmlcells_face1 = bc2D.get_pmlcells_face1();
		_pmlcells_face2 = bc2D.get_pmlcells_face2();
		_pmlcells_face3 = bc2D.get_pmlcells_face3();
		_pmlcells_face4 = bc2D.get_pmlcells_face4();
		_geom2D = std::move(bc2D._geom2D);
		_data1 = std::move(bc2D._data1);
		_data2 = std::move(bc2D._data2);
		_data3 = std::move(bc2D._data3);
		_data4 = std::move(bc2D._data4);
		return *this;
	}

	void BoundaryCondition2D::set_bc_face1(const std::string &bc_face1) {
		if (this->is_valid_boundary_condition(bc_face1)) {
			_bc_face1 = bc_face1;
			if (bc_face1 == "PML") {
				_data1.clear();
				_data1.shrink_to_fit();
			}
		}
		else {
			std::cerr << "Not a valid boundary condition. Boundary condition not changed." << std::endl;
		}
	}

	void BoundaryCondition2D::set_bc_face2(const std::string &bc_face2) {
		if (this->is_valid_boundary_condition(bc_face2)) {
			_bc_face2 = bc_face2;
			if (bc_face2 == "PML") {
				_data2.clear();
				_data2.shrink_to_fit();
			}
		}
		else {
			std::cerr << "Not a valid boundary condition. Boundary condition not changed." << std::endl;
		}
	}

	void BoundaryCondition2D::set_bc_face3(const std::string &bc_face3) {
		if (this->is_valid_boundary_condition(bc_face3)) {
			_bc_face3 = bc_face3;
			if (bc_face3 == "PML") {
				_data3.clear();
				_data3.shrink_to_fit();
			}
		}
		else {
			std::cerr << "Not a valid boundary condition. Boundary condition not changed." << std::endl;
		}
	}

	void BoundaryCondition2D::set_bc_face4(const std::string &bc_face4) {
		if (this->is_valid_boundary_condition(bc_face4)) {
			_bc_face4 = bc_face4;
			if (bc_face4 == "PML") {
				_data4.clear();
				_data4.shrink_to_fit();
			}
		}
		else {
			std::cerr << "Not a valid boundary condition. Boundary condition not changed." << std::endl;
		}
	}

	void BoundaryCondition2D::set_pmlcells_face1(const size_t &pmlcells_face1) {
		if (pmlcells_face1 >= Godzilla::PML_CELLS_DEFAULT) {
			_pmlcells_face1 = (_bc_face1 == "PML") ? pmlcells_face1 : _pmlcells_face1;
		}
		else {
			std::cerr << "Number of PML cells cannot be less than default. PML cells unchanged." << std::endl;
		}
	}

	void BoundaryCondition2D::set_pmlcells_face2(const size_t &pmlcells_face2) {
		if (pmlcells_face2 >= Godzilla::PML_CELLS_DEFAULT) {
			_pmlcells_face2 = (_bc_face2 == "PML") ? pmlcells_face2 : _pmlcells_face2;
		}
		else {
			std::cerr << "Number of PML cells cannot be less than default. PML cells unchanged." << std::endl;
		}
	}

	void BoundaryCondition2D::set_pmlcells_face3(const size_t &pmlcells_face3) {
		if (pmlcells_face3 >= Godzilla::PML_CELLS_DEFAULT) {
			_pmlcells_face3 = (_bc_face3 == "PML") ? pmlcells_face3 : _pmlcells_face3;
		}
		else {
			std::cerr << "Number of PML cells cannot be less than default. PML cells unchanged." << std::endl;
		}
	}

	void BoundaryCondition2D::set_pmlcells_face4(const size_t &pmlcells_face4) {
		if (pmlcells_face4 >= Godzilla::PML_CELLS_DEFAULT) {
			_pmlcells_face4 = (_bc_face4 == "PML") ? pmlcells_face4 : _pmlcells_face4;
		}
		else {
			std::cerr << "Number of PML cells cannot be less than default. PML cells unchanged." << std::endl;
		}
	}

	void BoundaryCondition2D::set_data(const Godzilla::vecxd &data1, const Godzilla::vecxd &data2,
									   const Godzilla::vecxd &data3, const Godzilla::vecxd &data4) {

		bool flag = true;
		if ((!this->is_PML(_bc_face1)) && (data1.size() != _geom2D.get_nY())) flag = false;
		if ((!this->is_PML(_bc_face2)) && (data2.size() != _geom2D.get_nX())) flag = false;
		if ((!this->is_PML(_bc_face3)) && (data3.size() != _geom2D.get_nY())) flag = false;
		if ((!this->is_PML(_bc_face4)) && (data4.size() != _geom2D.get_nX())) flag = false;

		if (flag == true) {
			if (this->is_data_consistent(data1, data2, data3, data4)) {
				if (!this->is_PML(_bc_face1)) _data1 = data1;
				if (!this->is_PML(_bc_face2)) _data2 = data2;
				if (!this->is_PML(_bc_face3)) _data3 = data3;
				if (!this->is_PML(_bc_face4)) _data4 = data4;
			}
			else {
				std::cerr << "Boundary condition not consistent. Data unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Dimensions of input data mismatch with geometry. Data unchanged." << std::endl;
		}
	}

	// Private methods
	bool BoundaryCondition2D::is_valid_boundary_condition(const std::string &bcface) const {
		if (bcface == "DBC") return true;
		if (bcface == "NBC") return true;
		if (bcface == "PML") return true;
		return false;
	}

	bool BoundaryCondition2D::is_data_consistent(const Godzilla::vecxd &data1, const Godzilla::vecxd &data2,
												 const Godzilla::vecxd &data3, const Godzilla::vecxd &data4) const {

		size_t n1 = _geom2D.get_nX() - 1;
		size_t n2 = _geom2D.get_nY() - 1;
		bool flag = true;
		if (_bc_face1 == "DBC") {
			if ((_bc_face2 == "DBC") && (data1[n2] != data2[0])) flag = false;
			if ((_bc_face4 == "DBC") && (data1[0] != data4[0])) flag = false;
		}
		if (_bc_face3 == "DBC") {
			if ((_bc_face2 == "DBC") && (data3[n2] != data2[n1])) flag = false;
			if ((_bc_face4 == "DBC") && (data3[0] != data4[n1])) flag = false;
		}
		return flag;
	}
}