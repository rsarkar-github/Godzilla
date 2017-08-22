#include "velocity1D.h"
#include <cmath>

namespace Godzilla {
	// Constructors
	Velocity1D::Velocity1D() : _geom1D(Godzilla::Geometry1D()), _name(""), _lock(false), _lockptr(nullptr) {
		_data.assign(_geom1D.get_nX(), 1.);
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const std::string &name) : _geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {
		_data.assign(geom1D.get_nX(), 1.);
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const double &scalar, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {
		
		double v = std::abs(scalar);
		if ((v >= Godzilla::VEL_MIN_TOL) && (v <= Godzilla::VEL_MAX_TOL)) {
			_data.assign(geom1D.get_nX(), scalar);
		}
		else {
			_data.assign(geom1D.get_nX(), 1.);
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
			_data.assign(geom1D.get_nX(), 1.);
		}
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecd &data, const std::string &name) :
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
		_data.assign(n, 1.);
		if (flag) {
			const double *data_ptr = data.data();
			Godzilla::xd *_data_ptr = _data.data();
			for (size_t i = 0; i < n; ++i) {
				_data_ptr[i] = data_ptr[i];			// Assigns the reals and sets complex part to zero
			}
		}
		_data.shrink_to_fit();
	}

	Velocity1D::Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecxd &data, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom1D.get_nX();
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
			_data.assign(n, 1.);
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

	void Velocity1D::set_data(const Godzilla::vecd &data) {
		if (!_lock) {
			size_t n = data.size();
			bool flag = (this->get_nelem() == n) ? true : false;
			if (flag) {
				const double *data_ptr = data.data();
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
				const double *data_ptr = data.data();
				Godzilla::xd *_data_ptr = _data.data();
				for (size_t i = 0; i < n; ++i) {
					_data_ptr[i] = data_ptr[i];			// Assigns the reals and sets complex part to zero
				}
			}
			else {
				std::cerr << "Data size does not match or data contains values too big or too small. Velocity 1D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Velocity1D locked. Velocity1D unchanged." << std::endl;
		}
	}

	void Velocity1D::set_data(const Godzilla::vecxd &data) {
		if (!_lock) {
			size_t n = data.size();
			bool flag = (this->get_nelem() == n) ? true : false;
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
				std::cerr << "Data size does not match or data contains values too big or too small. Velocity 1D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Velocity1D locked. Velocity1D unchanged." << std::endl;
		}
	}

	void Velocity1D::setmove_data(Godzilla::vecxd &data) {
		if (!_lock) {
			size_t n = data.size();
			bool flag = (this->get_nelem() == n) ? true : false;
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
				_data = std::move(data);
			}
			else {
				std::cerr << "Data size does not match or data contains values too big or too small. Velocity 1D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Velocity1D locked. Velocity1D unchanged." << std::endl;
		}
	}

	void Velocity1D::set_name(const std::string &name) {
		if (!_lock) {
			_name = name;
		}
		else {
			std::cerr << "Velocity1D locked. Velocity1D unchanged." << std::endl;
		}
	}

	void Velocity1D::activate_lock(waveX::LockManager<Godzilla::Velocity1D, Godzilla::vecxd> *lock) {
		if (!_lock) {
			if (lock->get_id() == Godzilla::VELOCITY1D_DATA_ID) {
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

	bool Velocity1D::is_equal(const Godzilla::Velocity1D &vel1D, const bool &name_except) const {
		if (name_except) {
			return (_geom1D.is_equal(vel1D.get_geom1D(), false) && (_data == vel1D.get_cdata())) ? true : false;
		}
		else {
			return (_geom1D.is_equal(vel1D.get_geom1D(), true) && (_data == vel1D.get_cdata()) && (_name == vel1D.get_name())) ? true : false;
		}
	}
}