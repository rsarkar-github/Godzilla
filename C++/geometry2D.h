#pragma once
#pragma once
#include "axis.h"

namespace Godzilla {
	class Geometry2D {
	public:
		// Constructors
		Geometry2D() : _nX(2), _ncellsX(1), _startX(0.), _endX(1.), _lenX(1.), _hX(1.),
					   _nY(2), _ncellsY(1), _startY(0.), _endY(1.), _lenY(1.), _hY(1.) {}
		Geometry2D(const size_t &nX, const double &startX, const double &hX,
				   const size_t &nY, const double &startY, const double &hY);
		Geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
				   const double &startY, const double &endY, const size_t &ncellsY);
		Geometry2D(const waveX::Axis &axis2D);
		Geometry2D(const Godzilla::Geometry2D &geom2D);
		Geometry2D(Godzilla::Geometry2D &&geom2D);

		// Public methods
		Godzilla::Geometry2D& operator=(const Godzilla::Geometry2D &geom2D);
		Godzilla::Geometry2D& operator=(Godzilla::Geometry2D &&geom2D);
		size_t get_nX() const { return _nX; }
		size_t get_ncellsX() const { return _ncellsX; }
		double get_startX() const { return _startX; }
		double get_endX() const { return _endX; }
		double get_lenX() const { return _lenX; }
		double get_hX() const { return _hX; }
		size_t get_nY() const { return _nY; }
		size_t get_ncellsY() const { return _ncellsY; }
		double get_startY() const { return _startY; }
		double get_endY() const { return _endY; }
		double get_lenY() const { return _lenY; }
		double get_hY() const { return _hY; }
		void get_geometry2D(size_t *nX, size_t *ncellsX, double *startX, double *endX, double *lenX, double *hX,
							size_t *nY, size_t *ncellsY, double *startY, double *endY, double *lenY, double *hY) const;

		void set_geometry2D(const size_t &nX, const double &startX, const double &hX,
							const size_t &nY, const double &startY, const double &hY);
		void set_geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
							const double &startY, const double &endY, const size_t &ncellsY);
		void set_geometry2D(const waveX::Axis &axis2D);

	private:
		// Private members
		size_t _nX;             // id = 0
		size_t _ncellsX;        // id = 1
		double _startX;         // id = 2
		double _endX;           // id = 3
		double _lenX;           // id = 4
		double _hX;             // id = 5

		size_t _nY;             // id = 6
		size_t _ncellsY;        // id = 7
		double _startY;         // id = 8
		double _endY;           // id = 9
		double _lenY;           // id = 10
		double _hY;             // id = 11
	};
}