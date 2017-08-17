#include "velocity1D.h"
#include <cmath>

namespace Godzilla {
	// Constructors
	Velocity1D::Velocity1D() : _geom1D(Godzilla::Geometry1D()), _name(""), _lock(false), _lockptr(nullptr) {
		_data.assign(_geom1D.get_nX(), (1., 0.));
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const std::string &name) : _geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {
		_data.assign(geom1D.get_nX(), (1., 0.));
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const double &scalar, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {
		
		double v = std::abs(scalar);
		if ((v >= Godzilla::VEL_MIN_TOL) && (v <= Godzilla::VEL_MAX_TOL)) {
			_data.assign(geom1D.get_nX(), (scalar, 0.));
		}
		else {
			_data.assign(geom1D.get_nX(), (1., 0.));
		}
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::xd &scalar, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {
		
		double v = std::abs(scalar);
		if ((v >= Godzilla::VEL_MIN_TOL) && (v <= Godzilla::VEL_MAX_TOL)) {
			_data.assign(geom1D.get_nX(), scalar);
		}
		else {
			_data.assign(geom1D.get_nX(), (1., 0.));
		}
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecd &data, const std::string &name = "") :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom1D.get_nX();
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

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecxd &data, const std::string &name = "") :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom1D.get_nX();
		bool flag = (data.size() == n) ? true : false;
		if (flag) {
			const Godzilla::xd *ptr = data.data();
			double v = 0;
			for (size_t i = 0; i < n; ++i) {
				v = std::abs(ptr[i]);
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

	Velocity1D::Velocity1D(const Godzilla::Velocity1D &vel1D) : _lock(false), _lockptr(nullptr) {
		_geom1D = vel1D.get_geom1D();
		_data = vel1D.get_cdata();
		_name = vel1D.get_name();
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(Godzilla::Velocity1D &&vel1D) : _lock(false), _lockptr(nullptr) {
		_geom1D = vel1D.get_geom1D();
		_data = std::move(vel1D._data);
		_name = std::move(vel1D._name);
	}

	// Public methods
	Godzilla::Velocity1D& Velocity1D::operator=(const Godzilla::Velocity1D &vel1D) {
		if (!_lock) {
			_geom1D = vel1D.get_geom1D();
			_data = vel1D.get_cdata();
			_name = vel1D.get_name();
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Velocity1D locked. Velocity1D unchanged." << std::endl;
			return *this;
		}
		_data.shrink_to_fit();
	}

	Godzilla::Velocity1D& Velocity1D::operator=(Godzilla::Velocity1D &&vel1D) {
		if (!_lock) {
			_geom1D = vel1D.get_geom1D();
			_data = std::move(vel1D._data);
			_name = std::move(vel1D._name);
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Velocity1D locked. Velocity1D unchanged." << std::endl;
			return *this;
		}
	}

	void Velocity1D::activate_lock(waveX::LockManager<Godzilla::Velocity1D, Godzilla::vecxd> *lock) {
		if (!_lock) {
			if (lock->get_id() == 1) {
				lock->_ptr = &_data;
				_lock = true;
				_lockptr = lock;
			}
		}
	}

	void Velocity1D::deactivate_lock(waveX::LockManager<Godzilla::Velocity1D, Godzilla::vecxd> *lock) {
		if (_lock && (lock == _lockptr)) {
			lock->_ptr = nullptr;
			_lock = false;
			_lockptr = nullptr;
		}
	}

}