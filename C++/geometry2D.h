#pragma once
#include "axis.h"

namespace Godzilla {
	class Geometry2D {
		public:
			// Constructors
			Geometry2D() : _nX(2), _ncellsX(1), _startX(0.), _endX(1.), _lenX(1.), _hX(1.), _labelX(""),
						   _nY(2), _ncellsY(1), _startY(0.), _endY(1.), _lenY(1.), _hY(1.), _labelY("") {}
			Geometry2D(const size_t &nX, const double &startX, const double &hX,
					   const size_t &nY, const double &startY, const double &hY, const std::string &labelX = "", const std::string &labelY = "");
			Geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
					   const double &startY, const double &endY, const size_t &ncellsY, const std::string &labelX = "", const std::string &labelY = "");
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
			std::string get_labelX() const { return _labelX; }
			size_t get_nY() const { return _nY; }
			size_t get_ncellsY() const { return _ncellsY; }
			double get_startY() const { return _startY; }
			double get_endY() const { return _endY; }
			double get_lenY() const { return _lenY; }
			double get_hY() const { return _hY; }
			std::string get_labelY() const { return _labelY; }
			void get_geometry2D(size_t *nX, size_t *ncellsX, double *startX, double *endX, double *lenX, double *hX, std::string *labelX,
								size_t *nY, size_t *ncellsY, double *startY, double *endY, double *lenY, double *hY, std::string *labelY) const;

			void set_geometry2D(const size_t &nX, const double &startX, const double &hX, const size_t &nY, const double &startY, const double &hY);
			void set_geometry2D(const size_t &nX, const double &startX, const double &hX,
								const size_t &nY, const double &startY, const double &hY, const std::string &labelX, const std::string &labelY);
			void set_geometry2D(const double &startX, const double &endX, const size_t &ncellsX, const double &startY, const double &endY, const size_t &ncellsY);
			void set_geometry2D(const double &startX, const double &endX, const size_t &ncellsX,
								const double &startY, const double &endY, const size_t &ncellsY, const std::string &labelX, const std::string &labelY);
			void set_geometry2D(const waveX::Axis &axis2D);

			bool is_equal(const Godzilla::Geometry2D &geom2D, const bool &name_except = true) const;

		private:
			// Private members
			size_t _nX;             // id = 0
			size_t _ncellsX;        // id = 1
			double _startX;         // id = 2
			double _endX;           // id = 3
			double _lenX;           // id = 4
			double _hX;             // id = 5
			std::string _labelX;    // id = 6

			size_t _nY;             // id = 7
			size_t _ncellsY;        // id = 8
			double _startY;         // id = 9
			double _endY;           // id = 10
			double _lenY;           // id = 11
			double _hY;             // id = 12
			std::string _labelY;    // id = 13
	};
}