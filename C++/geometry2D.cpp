#include "geometry2D.h"
#include <iostream>
#include <cmath>

namespace Godzilla {
	// Constructors

	Geometry2D::Geometry2D(const size_t &nX, const double &startX, const double &hX,
						   const size_t &nY, const double &startY, const double &hY) {

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
			std::cout << "(nX > 1) && (hX != 0.) && (nY > 1) && (hY != 0.) not satisfied." << std::endl;

			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;

			_nY = 2;
			_ncellsY = 1;
			_startY = 0.;
			_endY = 1.;
			_lenY = 1.;
			_hY = 1.;
		}
	}

	Geometry2D::Geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
						   const double &startY, const double &endY, const size_t &ncellsY) {

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
			std::cout << "(ncellsX > 0) && (startX != endX) && (ncellsY > 0) && (startY != endY) not satisfied" << std::endl;

			_nX = 2;
			_ncellsX = 1;
			_startX = 0.;
			_endX = 1.;
			_lenX = 1.;
			_hX = 1.;

			_nY = 2;
			_ncellsY = 1;
			_startY = 0.;
			_endY = 1.;
			_lenY = 1.;
			_hY = 1.;
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

			_nY = axis2D.get_n()[1];
			_ncellsY = _nY - 1;
			_hY = axis2D.get_d()[1];
			const double pY = _ncellsY * _hY;
			_startY = axis2D.get_o()[1];
			_endY = _startY + pY;
			_lenY = (pY > 0) ? pY : -pY;
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

			_nY = 2;
			_ncellsY = 1;
			_startY = 0.;
			_endY = 1.;
			_lenY = 1.;
			_hY = 1.;
		}
	}

	Geometry2D::Geometry2D(const Godzilla::Geometry2D &geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY);
	}

	Geometry2D::Geometry2D(Godzilla::Geometry2D &&geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY);
	}


	// Public methods
	Godzilla::Geometry2D& Geometry2D::operator=(const Godzilla::Geometry2D &geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY);
	}

	Godzilla::Geometry2D& Geometry2D::operator=(Godzilla::Geometry2D &&geom2D) {
		geom2D.get_geometry2D(&_nX, &_ncellsX, &_startX, &_endX, &_lenX, &_hX, &_nY, &_ncellsY, &_startY, &_endY, &_lenY, &_hY);
	}

	void Geometry2D::get_geometry2D(size_t *nX, size_t *ncellsX, double *startX, double *endX, double *lenX, double *hX,
									size_t *nY, size_t *ncellsY, double *startY, double *endY, double *lenY, double *hY) const {
		*nX = _nX;
		*ncellsX = _ncellsX;
		*startX = _startX;
		*endX = _endX;
		*lenX = _lenX;
		*hX = _hX;

		*nY = _nY;
		*ncellsY = _ncellsY;
		*startY = _startY;
		*endY = _endY;
		*lenY = _lenY;
		*hY = _hY;
	}

	void Geometry2D::set_geometry2D(const size_t &nX, const double &startX, const double &hX,
									const size_t &nY, const double &startY, const double &hY) {

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

	void Geometry2D::set_geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
									const double &startY, const double &endY, const size_t &ncellsY) {

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

	void Geometry2D::set_geometry2D(const waveX::Axis &axis2D) {
		if ((axis2D.get_ndims() == 2) && (axis2D.get_n()[0] > 1) && (axis2D.get_d()[0] != 0.) && (axis2D.get_n()[1] > 1) && (axis2D.get_d()[1] != 0.)) {
			_nX = axis2D.get_n()[0];
			_ncellsX = _nX - 1;
			_hX = axis2D.get_d()[0];
			const double pX = _ncellsX * _hX;
			_startX = axis2D.get_o()[0];
			_endX = _startX + pX;
			_lenX = (pX > 0) ? pX : -pX;

			_nY = axis2D.get_n()[1];
			_ncellsY = _nY - 1;
			_hY = axis2D.get_d()[1];
			const double pY = _ncellsY * _hY;
			_startY = axis2D.get_o()[1];
			_endY = _startY + pY;
			_lenY = (pY > 0) ? pY : -pY;
		}
		else {
			std::cout << "(axis2D.get_ndims() == 2) && (axis2D.get_n()[0] > 1) && (axis2D.get_d()[0] != 0.)"
					  << "&& (axis2D.get_n()[1] > 1) && (axis2D.get_d()[1] != 0.) not satisfied. No modification done" << std::endl;
		}
	}
}