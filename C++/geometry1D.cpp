#include "geometry1D.h"
#include <iostream>
#include <cmath>

namespace Godzilla {
	// Constructors
	Geometry1D::Geometry1D(const size_t &nX, const double &startX, const double &hX, const std::string &labelX) {
		if ((nX > 1) && (hX != 0.)) {
			_nX = nX;
			_ncellsX = nX - 1;
			_hX = hX;
			const double pX = _ncellsX * hX;
			_startX = startX;
			_endX = startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;
		}
		else {
			std::cout << "(nX > 1) && (hX != 0.) not satisfied." << std::endl;
			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;
			_labelX = "";
		}
	}

	Geometry1D::Geometry1D(const double &startX, const double &endX, const size_t &ncellsX, const std::string &labelX) {
		if ((ncellsX > 0) && (startX != endX)) {
			_nX = ncellsX + 1;
			_ncellsX = ncellsX;
			_startX = startX;
			_endX = endX;
			const double pX = endX - startX;
			_hX = pX / ncellsX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;
		}
		else {
			std::cout << "(ncellsX > 0) && (startX != endX) not satisfied" << std::endl;
			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;
			_labelX = "";
		}
	}

	Geometry1D::Geometry1D(const waveX::Axis &axis1D) {
		if ((axis1D.get_ndims() == 1) && (axis1D.get_n()[0] > 1) && (axis1D.get_d()[0] != 0.)) {
			_nX = axis1D.get_n()[0];
			_ncellsX = _nX - 1;
			_hX = axis1D.get_d()[0];
			const double pX = _ncellsX * _hX;
			_startX = axis1D.get_o()[0];
			_endX = _startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = axis1D.get_label()[0];
		}
		else {
			std::cout << "(axis1D.get_ndims() == 1) && (axis1D.get_n()[0] > 1) && (axis1D.get_d()[0] != 0.) not satisfied" << std::endl;
			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;
			_labelX = "";
		}
	}

	Geometry1D::Geometry1D(const Godzilla::Geometry1D &geom1D) {
		geom1D.get_geometry1D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX);
	}

	Geometry1D::Geometry1D(Godzilla::Geometry1D &&geom1D) {
		geom1D.get_geometry1D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX);
	}

	// Public methods
	Godzilla::Geometry1D& Geometry1D::operator=(const Godzilla::Geometry1D &geom1D) {
		geom1D.get_geometry1D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX);
		return *this;
	}

	Godzilla::Geometry1D& Geometry1D::operator=(Godzilla::Geometry1D &&geom1D) {
		geom1D.get_geometry1D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX);
		return *this;
	}

	void Geometry1D::get_geometry1D(size_t *nX, size_t *ncellsX, double *startX, double *endX, double *lenX, double *hX, std::string *labelX) const {
		*nX = _nX;
		*ncellsX = _ncellsX;
		*startX = _startX;
		*endX = _endX;
		*lenX = _lenX;
		*hX = _hX;
		*labelX = _labelX;
	}

	void Geometry1D::set_geometry1D(const size_t &nX, const double &startX, const double &hX, const std::string &labelX) {
		if ((nX > 1) && (hX != 0.)) {
			_nX = nX;
			_ncellsX = nX - 1;
			_hX = hX;
			const double pX = _ncellsX * hX;
			_startX = startX;
			_endX = startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;
		}
		else {
			std::cout << "(nX > 1) && (hX != 0.) not satisfied. No modification done" << std::endl;
		}
	}

	void Geometry1D::set_geometry1D(const double &startX, const double &endX, const size_t &ncellsX, const std::string &labelX) {
		if ((ncellsX > 0) && (startX != endX)) {
			_nX = ncellsX + 1;
			_ncellsX = ncellsX;
			_startX = startX;
			_endX = endX;
			const double pX = endX - startX;
			_hX = pX / ncellsX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;
		}
		else {
			std::cout << "(ncellsX > 0) && (startX != endX) not satisfied. No modification done" << std::endl;
		}
	}

	void Geometry1D::set_geometry1D(const waveX::Axis &axis1D) {
		if ((axis1D.get_ndims() == 1) && (axis1D.get_n()[0] > 1) && (axis1D.get_d()[0] != 0.)) {
			_nX = axis1D.get_n()[0];
			_ncellsX = _nX - 1;
			_hX = axis1D.get_d()[0];
			const double pX = _ncellsX * _hX;
			_startX = axis1D.get_o()[0];
			_endX = _startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = axis1D.get_label()[0];
		}
		else {
			std::cout << "(axis1D.get_ndims() == 1) && (axis1D.get_n()[0] > 1) && (axis1D.get_d()[0] != 0.) not satisfied."
					  << "No modification done." << std::endl;
		}
	}

	bool Geometry1D::is_equal(const Godzilla::Geometry1D &geom1D, const bool &name_except) const {
		if (name_except) {
			if ((_nX != geom1D.get_nX()) || (_ncellsX != geom1D.get_ncellsX())) return false;
			if ((_startX != geom1D.get_startX()) || (_endX != geom1D.get_endX())) return false;
			if ((_lenX != geom1D.get_lenX()) || (_hX != geom1D.get_hX())) return false;
			return true;
		}
		else {
			if ((_nX != geom1D.get_nX()) || (_ncellsX != geom1D.get_ncellsX())) return false;
			if ((_startX != geom1D.get_startX()) || (_endX != geom1D.get_endX())) return false;
			if ((_lenX != geom1D.get_lenX()) || (_hX != geom1D.get_hX()) || (_labelX != geom1D.get_labelX())) return false;
			return true;
		}
	}

}