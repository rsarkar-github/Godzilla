#include "velocity2D.h"
#include <cmath>

namespace Godzilla {
	// Constructors
	Velocity2D::Velocity2D() : _geom2D(Godzilla::Geometry2D()), _name(""), _lock(false), _lockptr(nullptr) {
		_data.assign(_geom2D.get_nX() * _geom2D.get_nY(), (1., 0.));
		_data.shrink_to_fit();
	}

	Velocity2D::Velocity2D(const Godzilla::Geometry2D &geom2D, const std::string &name) : _geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {
		_data.assign(geom2D.get_nX() * geom2D.get_nY(), (1., 0.));
		_data.shrink_to_fit();
	}

	Velocity2D::Velocity2D(const Godzilla::Geometry2D &geom2D, const double &scalar, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		double v = std::abs(scalar);
		if ((v >= Godzilla::VEL_MIN_TOL) && (v <= Godzilla::VEL_MAX_TOL)) {
			_data.assign(geom2D.get_nX() * geom2D.get_nY(), (scalar, 0.));
		}
		else {
			_data.assign(geom2D.get_nX() * geom2D.get_nY(), (1., 0.));
		}
		_data.shrink_to_fit();
	}

	Velocity2D::Velocity2D(const Godzilla::Geometry2D &geom2D, const Godzilla::xd &scalar, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		double v = std::abs(scalar);
		if ((v >= Godzilla::VEL_MIN_TOL) && (v <= Godzilla::VEL_MAX_TOL)) {
			_data.assign(geom2D.get_nX() * geom2D.get_nY(), scalar);
		}
		else {
			_data.assign(geom2D.get_nX() * geom2D.get_nY(), (1., 0.));
		}
		_data.shrink_to_fit();
	}

	Velocity2D::Velocity2D(const Godzilla::Geometry2D &geom2D, const Godzilla::vecd &data, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom2D.get_nX() * geom2D.get_nY();
		bool flag = (data.size() == n) ? true : false;
		if (flag) {
			const double *data_ptr = data.data();
			double v = 0.;
			for (size_t i = 0; i < n; ++i) {
				v = std::abs(data_ptr[i]);
				if ((v < Godzilla::VEL_MIN_TOL) || (v > Godzilla::VEL_MAX_TOL)) {
					flag = false;
					break;
				}
			}
		}

		if (flag) {
			_data.reserve(n);
			const double *data_ptr = data.data();
			Godzilla::xd *_data_ptr = _data.data();
			for (size_t i = 0; i < n; ++i) {
				_data_ptr[i] = data_ptr[i];			// Assigns the reals and sets complex part to zero
			}
		}
		else {
			_data.assign(n, (1., 0.));
		}
		_data.shrink_to_fit();
	}

	Velocity2D::Velocity2D(const Godzilla::Geometry2D &geom2D, const Godzilla::vecxd &data, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom2D.get_nX() * geom2D.get_nY();
		bool flag = (data.size() == n) ? true : false;
		if (flag) {
			const Godzilla::xd *data_ptr = data.data();
			double v = 0;
			for (size_t i = 0; i < n; ++i) {
				v = std::abs(data_ptr[i]);
				if ((v < Godzilla::VEL_MIN_TOL) || (v > Godzilla::VEL_MAX_TOL)) {
					flag = false;
					break;
				}
			}
		}

		if (flag) {
			_data = data;
		}
		else {
			_data.assign(n, (1., 0.));
		}
		_data.shrink_to_fit();
	}

	Velocity2D::Velocity2D(const Godzilla::Velocity2D &vel2D) : _lock(false), _lockptr(nullptr) {
		_geom2D = vel2D.get_geom2D();
		_data = vel2D.get_cdata();
		_name = vel2D.get_name();
		_data.shrink_to_fit();
	}

	Velocity2D::Velocity2D(Godzilla::Velocity2D &&vel2D) : _lock(false), _lockptr(nullptr) {
		_geom2D = vel2D.get_geom2D();
		_data = std::move(vel2D._data);
		_name = std::move(vel2D._name);
	}

	// Public methods
	Godzilla::Velocity2D& Velocity2D::operator=(const Godzilla::Velocity2D &vel2D) {
		if (!_lock) {
			_geom2D = vel2D.get_geom2D();
			_data = vel2D.get_cdata();
			_name = vel2D.get_name();
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Velocity2D locked. Velocity2D unchanged." << std::endl;
			return *this;
		}
		_data.shrink_to_fit();
	}

	Godzilla::Velocity2D& Velocity2D::operator=(Godzilla::Velocity2D &&vel2D) {
		if (!_lock) {
			_geom2D = vel2D.get_geom2D();
			_data = std::move(vel2D._data);
			_name = std::move(vel2D._name);
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Velocity2D locked. Velocity2D unchanged." << std::endl;
			return *this;
		}
	}

	void Velocity2D::activate_lock(waveX::LockManager<Godzilla::Velocity2D, Godzilla::vecxd> *lock) {
		if (!_lock) {
			if (lock->get_id() == Godzilla::VELOCITY2D_DATA_ID) {
				lock->_ptr = &_data;
				_lock = true;
				_lockptr = lock;
			}
		}
	}

	void Velocity2D::deactivate_lock(waveX::LockManager<Godzilla::Velocity2D, Godzilla::vecxd> *lock) {
		if (_lock && (lock == _lockptr)) {
			lock->_ptr = nullptr;
			_lock = false;
			_lockptr = nullptr;
		}
	}
}