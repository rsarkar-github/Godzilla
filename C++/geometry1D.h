#pragma once
#include "axis.h"

namespace Godzilla {
	class Geometry1D {
		public:
			// Constructors
			Geometry1D() : _nX(2), _ncellsX(1), _startX(0.), _endX(1.), _lenX(1.), _hX(1.) {}
			Geometry1D(const size_t &nX, const double &startX, const double &hX);
			Geometry1D(const double &startX, const double &endX, const size_t &ncellsX);
			Geometry1D(const waveX::Axis &axis1D);
			Geometry1D(const Godzilla::Geometry1D &geom1D);
			Geometry1D(Godzilla::Geometry1D &&geom1D);

			// Public methods
			Godzilla::Geometry1D& operator=(const Godzilla::Geometry1D &geom1D);
			Godzilla::Geometry1D& operator=(Godzilla::Geometry1D &&geom1D);
			size_t get_nX() const { return _nX; }
			size_t get_ncellsX() const { return _ncellsX; }
			double get_startX() const { return _startX; }
			double get_endX() const { return _endX; }
			double get_lenX() const { return _lenX; }
			double get_hX() const { return _hX; }
			void get_geometry1D(size_t *nX, size_t *ncellsX, double *startX, double *endX, double *lenX, double *hX) const;

			void set_geometry1D(const size_t &nX, const double &startX, const double &hX);
			void set_geometry1D(const double &startX, const double &endX, const size_t &ncellsX);
			void set_geometry1D(const waveX::Axis &axis1D);

		private:
			// Private members
			size_t _nX;             // id = 0
			size_t _ncellsX;        // id = 1
			double _startX;         // id = 2
			double _endX;           // id = 3
			double _lenX;           // id = 4
			double _hX;             // id = 5
	};
}