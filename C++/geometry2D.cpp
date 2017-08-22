#include "geometry2D.h"
#include <iostream>
#include <cmath>

namespace Godzilla {
	// Constructors
	Geometry2D::Geometry2D(const size_t &nX, const double &startX, const double &hX,
						   const size_t &nY, const double &startY, const double &hY, const std::string &labelX, const std::string &labelY) {

		if ((nX > 1) && (hX != 0.) && (nY > 1) && (hY != 0.)) {
			_nX = nX;
			_ncellsX = nX - 1;
			_hX = hX;
			const double pX = _ncellsX * hX;
			_startX = startX;
			_endX = startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;

			_nY = nY;
			_ncellsY = nY - 1;
			_hY = hY;
			const double pY = _ncellsY * hY;
			_startY = startY;
			_endY = startY + pY;
			_lenY = (pY > 0) ? pY : -pY;
			_labelY = labelY;
		}
		else {
			std::cout << "(nX > 1) && (hX != 0.) && (nY > 1) && (hY != 0.) not satisfied." << std::endl;

			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;
			_labelX = "";

			_nY = 2;
			_ncellsY = 1;
			_startY = 0.;
			_endY = 1.;
			_lenY = 1.;
			_hY = 1.;
			_labelY = "";
		}
	}

	Geometry2D::Geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
						   const double &startY, const double &endY, const size_t &ncellsY, const std::string &labelX, const std::string &labelY) {

		if ((ncellsX > 0) && (startX != endX) && (ncellsY > 0) && (startY != endY)) {
			_nX = ncellsX + 1;
			_ncellsX = ncellsX;
			_startX = startX;
			_endX = endX;
			const double pX = endX - startX;
			_hX = pX / ncellsX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;

			_nY = ncellsY + 1;
			_ncellsY = ncellsY;
			_startY = startY;
			_endY = endY;
			const double pY = endY - startY;
			_hY = pY / ncellsY;
			_lenY = (pY > 0) ? pY : -pY;
			_labelY = labelY;
		}
		else {
			std::cout << "(ncellsX > 0) && (startX != endX) && (ncellsY > 0) && (startY != endY) not satisfied" << std::endl;

			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;
			_labelX = "";

			_nY = 2;
			_ncellsY = 1;
			_startY = 0.;
			_endY = 1.;
			_lenY = 1.;
			_hY = 1.;
			_labelY = "";
		}
	}

	Geometry2D::Geometry2D(const waveX::Axis &axis2D) {
		if ((axis2D.get_ndims() == 2) && (axis2D.get_n()[0] > 1) && (axis2D.get_d()[0] != 0.) && (axis2D.get_n()[1] > 1) && (axis2D.get_d()[1] != 0.)) {
			_nX = axis2D.get_n()[0];
			_ncellsX = _nX - 1;
			_hX = axis2D.get_d()[0];
			const double pX = _ncellsX * _hX;
			_startX = axis2D.get_o()[0];
			_endX = _startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = axis2D.get_label()[0];

			_nY = axis2D.get_n()[1];
			_ncellsY = _nY - 1;
			_hY = axis2D.get_d()[1];
			const double pY = _ncellsY * _hY;
			_startY = axis2D.get_o()[1];
			_endY = _startY + pY;
			_lenY = (pY > 0) ? pY : -pY;
			_labelY = axis2D.get_label()[1];
		}
		else {
			std::cout << "(axis2D.get_ndims() == 2) && (axis2D.get_n()[0] > 1) && (axis2D.get_d()[0] != 0.)"
					  << "&& (axis2D.get_n()[1] > 1) && (axis2D.get_d()[1] != 0.) not satisfied" << std::endl;
			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;
			_labelX = "";

			_nY = 2;
			_ncellsY = 1;
			_startY = 0.;
			_endY = 1.;
			_lenY = 1.;
			_hY = 1.;
			_labelY = "";
		}
	}

	Geometry2D::Geometry2D(const Godzilla::Geometry2D &geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY, &_labelY);
	}

	Geometry2D::Geometry2D(Godzilla::Geometry2D &&geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY, &_labelY);
	}

	// Public methods
	Godzilla::Geometry2D& Geometry2D::operator=(const Godzilla::Geometry2D &geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY, &_labelY);
		return *this;
	}

	Godzilla::Geometry2D& Geometry2D::operator=(Godzilla::Geometry2D &&geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_labelX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY, &_labelY);
		return *this;
	}

	void Geometry2D::get_geometry2D(size_t *nX, size_t *ncellsX, double *startX, double *endX, double *lenX, double *hX, std::string *labelX,
									size_t *nY, size_t *ncellsY, double *startY, double *endY, double *lenY, double *hY, std::string *labelY) const {
		*nX = _nX;
		*ncellsX = _ncellsX;
		*startX = _startX;
		*endX = _endX;
		*lenX = _lenX;
		*hX = _hX;
		*labelX = _labelX;

		*nY = _nY;
		*ncellsY = _ncellsY;
		*startY = _startY;
		*endY = _endY;
		*lenY = _lenY;
		*hY = _hY;
		*labelY = _labelY;
	}

	void Geometry2D::set_geometry2D(const size_t &nX, const double &startX, const double &hX, const size_t &nY, const double &startY, const double &hY) {

		if ((nX > 1) && (hX != 0.) && (nY > 1) && (hY != 0.)) {
			_nX = nX;
			_ncellsX = nX - 1;
			_hX = hX;
			const double pX = _ncellsX * hX;
			_startX = startX;
			_endX = startX + pX;
			_lenX = (pX > 0) ? pX : -pX;

			_nY = nY;
			_ncellsY = nY - 1;
			_hY = hY;
			const double pY = _ncellsY * hY;
			_startY = startY;
			_endY = startY + pY;
			_lenY = (pY > 0) ? pY : -pY;
		}
		else {
			std::cout << "(nX > 1) && (hX != 0.) && (nY > 1) && (hY != 0.) not satisfied. No modification done" << std::endl;
		}
	}


	void Geometry2D::set_geometry2D(const size_t &nX, const double &startX, const double &hX,
									const size_t &nY, const double &startY, const double &hY, const std::string &labelX, const std::string &labelY) {

		if ((nX > 1) && (hX != 0.) && (nY > 1) && (hY != 0.)) {
			_nX = nX;
			_ncellsX = nX - 1;
			_hX = hX;
			const double pX = _ncellsX * hX;
			_startX = startX;
			_endX = startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;

			_nY = nY;
			_ncellsY = nY - 1;
			_hY = hY;
			const double pY = _ncellsY * hY;
			_startY = startY;
			_endY = startY + pY;
			_lenY = (pY > 0) ? pY : -pY;
			_labelY = labelY;
		}
		else {
			std::cout << "(nX > 1) && (hX != 0.) && (nY > 1) && (hY != 0.) not satisfied. No modification done" << std::endl;
		}
	}

	void Geometry2D::set_geometry2D(const double &startX, const double &endX, const size_t &ncellsX, const double &startY, const double &endY, const size_t &ncellsY) {

		if ((ncellsX > 0) && (startX != endX) && (ncellsY > 0) && (startY != endY)) {
			_nX = ncellsX + 1;
			_ncellsX = ncellsX;
			_startX = startX;
			_endX = endX;
			const double pX = endX - startX;
			_hX = pX / ncellsX;
			_lenX = (pX > 0) ? pX : -pX;

			_nY = ncellsY + 1;
			_ncellsY = ncellsY;
			_startY = startY;
			_endY = endY;
			const double pY = endY - startY;
			_hY = pY / ncellsY;
			_lenY = (pY > 0) ? pY : -pY;
		}
		else {
			std::cout << "(ncellsX > 0) && (startX != endX) && (ncellsY > 0) && (startY != endY) not satisfied. No modification done" << std::endl;
		}
	}


	void Geometry2D::set_geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
									const double &startY, const double &endY, const size_t &ncellsY, const std::string &labelX, const std::string &labelY) {

		if ((ncellsX > 0) && (startX != endX) && (ncellsY > 0) && (startY != endY)) {
			_nX = ncellsX + 1;
			_ncellsX = ncellsX;
			_startX = startX;
			_endX = endX;
			const double pX = endX - startX;
			_hX = pX / ncellsX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = labelX;

			_nY = ncellsY + 1;
			_ncellsY = ncellsY;
			_startY = startY;
			_endY = endY;
			const double pY = endY - startY;
			_hY = pY / ncellsY;
			_lenY = (pY > 0) ? pY : -pY;
			_labelY = labelY;
		}
		else {
			std::cout << "(ncellsX > 0) && (startX != endX) && (ncellsY > 0) && (startY != endY) not satisfied. No modification done" << std::endl;
		}
	}

	void Geometry2D::set_geometry2D(const waveX::Axis &axis2D) {
		if ((axis2D.get_ndims() == 2) && (axis2D.get_n()[0] > 1) && (axis2D.get_d()[0] != 0.) && (axis2D.get_n()[1] > 1) && (axis2D.get_d()[1] != 0.)) {
			_nX = axis2D.get_n()[0];
			_ncellsX = _nX - 1;
			_hX = axis2D.get_d()[0];
			const double pX = _ncellsX * _hX;
			_startX = axis2D.get_o()[0];
			_endX = _startX + pX;
			_lenX = (pX > 0) ? pX : -pX;
			_labelX = axis2D.get_label()[0];

			_nY = axis2D.get_n()[1];
			_ncellsY = _nY - 1;
			_hY = axis2D.get_d()[1];
			const double pY = _ncellsY * _hY;
			_startY = axis2D.get_o()[1];
			_endY = _startY + pY;
			_lenY = (pY > 0) ? pY : -pY;
			_labelY = axis2D.get_label()[1];
		}
		else {
			std::cout << "(axis2D.get_ndims() == 2) && (axis2D.get_n()[0] > 1) && (axis2D.get_d()[0] != 0.)"
					  << "&& (axis2D.get_n()[1] > 1) && (axis2D.get_d()[1] != 0.) not satisfied. No modification done" << std::endl;
		}
	}

	bool Geometry2D::is_equal(const Godzilla::Geometry2D &geom2D, const bool &name_except) const {
		if (name_except) {
			if ((_nX != geom2D.get_nX()) || (_ncellsX != geom2D.get_ncellsX())) return false;
			if ((_startX != geom2D.get_startX()) || (_endX != geom2D.get_endX())) return false;
			if ((_lenX != geom2D.get_lenX()) || (_hX != geom2D.get_hX())) return false;
			if ((_nY != geom2D.get_nY()) || (_ncellsY != geom2D.get_ncellsY())) return false;
			if ((_startY != geom2D.get_startY()) || (_endY != geom2D.get_endY())) return false;
			if ((_lenY != geom2D.get_lenY()) || (_hY != geom2D.get_hY())) return false;
			return true;
		}
		else {
			if ((_nX != geom2D.get_nX()) || (_ncellsX != geom2D.get_ncellsX())) return false;
			if ((_startX != geom2D.get_startX()) || (_endX != geom2D.get_endX())) return false;
			if ((_lenX != geom2D.get_lenX()) || (_hX != geom2D.get_hX()) || (_labelX != geom2D.get_labelX())) return false;
			if ((_nY != geom2D.get_nY()) || (_ncellsY != geom2D.get_ncellsY())) return false;
			if ((_startY != geom2D.get_startY()) || (_endY != geom2D.get_endY())) return false;
			if ((_lenY != geom2D.get_lenY()) || (_hY != geom2D.get_hY()) || (_labelY != geom2D.get_labelY())) return false;
			return true;
		}
	}

}