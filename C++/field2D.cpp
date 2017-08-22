#include "field2D.h"
#include <cmath>

namespace Godzilla {
	// Constructors
	Field2D::Field2D() : _geom2D(Godzilla::Geometry2D()), _name(""), _lock(false), _lockptr(nullptr) {
		_data.assign(_geom2D.get_nX() * _geom2D.get_nY(), 0.);
		_data.shrink_to_fit();
	}

	Field2D::Field2D(const Godzilla::Geometry2D &geom2D, const std::string &name) : _geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {
		_data.assign(geom2D.get_nX() * geom2D.get_nY(), 0.);
		_data.shrink_to_fit();
	}

	Field2D::Field2D(const Godzilla::Geometry2D &geom2D, const double &scalar, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		_data.assign(geom2D.get_nX() * geom2D.get_nY(), scalar);
		_data.shrink_to_fit();
	}

	Field2D::Field2D(const Godzilla::Geometry2D &geom2D, const Godzilla::xd &scalar, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		_data.assign(geom2D.get_nX() * geom2D.get_nY(), scalar);
		_data.shrink_to_fit();
	}

	Field2D::Field2D(const Godzilla::Geometry2D &geom2D, const Godzilla::vecd &data, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom2D.get_nX() * geom2D.get_nY();
		_data.assign(n, 0.);
		if (data.size() == n) {
			const double *data_ptr = data.data();
			Godzilla::xd *_data_ptr = _data.data();
			for (size_t i = 0; i < n; ++i) {
				_data_ptr[i] = data_ptr[i];			// Assigns the reals and sets complex part to zero
			}
		}
		_data.shrink_to_fit();
	}

	Field2D::Field2D(const Godzilla::Geometry2D &geom2D, const Godzilla::vecxd &data, const std::string &name) :
		_geom2D(geom2D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom2D.get_nX() * geom2D.get_nY();
		if (data.size() == n) {
			_data = data;
		}
		else {
			_data.assign(n, 0.);
		}
		_data.shrink_to_fit();
	}

	Field2D::Field2D(const Godzilla::Field2D &field2D) : _lock(false), _lockptr(nullptr) {
		_geom2D = field2D.get_geom2D();
		_data = field2D.get_cdata();
		_name = field2D.get_name();
		_data.shrink_to_fit();
	}

	Field2D::Field2D(Godzilla::Field2D &&field2D) : _lock(false), _lockptr(nullptr) {
		_geom2D = field2D.get_geom2D();
		_data = std::move(field2D._data);
		_name = std::move(field2D._name);
	}

	// Public methods
	Godzilla::Field2D& Field2D::operator=(const Godzilla::Field2D &field2D) {
		if (!_lock) {
			_geom2D = field2D.get_geom2D();
			_data = field2D.get_cdata();
			_name = field2D.get_name();
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Field2D locked. Field2D unchanged." << std::endl;
			return *this;
		}
		_data.shrink_to_fit();
	}

	Godzilla::Field2D& Field2D::operator=(Godzilla::Field2D &&field2D) {
		if (!_lock) {
			_geom2D = field2D.get_geom2D();
			_data = std::move(field2D._data);
			_name = std::move(field2D._name);
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Field2D locked. Field2D unchanged." << std::endl;
			return *this;
		}
	}

	void Field2D::set_data(const Godzilla::vecd &data) {
		if (!_lock) {
			size_t n = data.size();
			if (this->get_nelem() == n) {
				const double *data_ptr = data.data();
				Godzilla::xd *_data_ptr = _data.data();
				for (size_t i = 0; i < n; ++i) {
					_data_ptr[i] = data_ptr[i];			// Assigns the reals and sets complex part to zero
				}
			}
			else {
				std::cerr << "Data size does not match or data contains values too big or too small. Field2D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Field2D locked. Field2D unchanged." << std::endl;
		}
	}

	void Field2D::set_data(const Godzilla::vecxd &data) {
		if (!_lock) {
			size_t n = data.size();
			if (this->get_nelem() == n) {
				_data = data;
			}
			else {
				std::cerr << "Data size does not match or data contains values too big or too small. Field2D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Field2D locked. Field2D unchanged." << std::endl;
		}
	}

	void Field2D::setmove_data(Godzilla::vecxd &data) {
		if (!_lock) {
			size_t n = data.size();
			if (this->get_nelem() == n) {
				_data = std::move(data);
			}
			else {
				std::cerr << "Data size does not match or data contains values too big or too small. Field2D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Field2D locked. Field2D unchanged." << std::endl;
		}
	}

	void Field2D::set_name(const std::string &name) {
		if (!_lock) {
			_name = name;
		}
		else {
			std::cerr << "Field2D locked. Field2D unchanged." << std::endl;
		}
	}

	void Field2D::activate_lock(waveX::LockManager<Godzilla::Field2D, Godzilla::vecxd> *lock) {
		if (!_lock) {
			if (lock->get_id() == Godzilla::FIELD2D_DATA_ID) {
				lock->_ptr = &_data;
				_lock = true;
				_lockptr = lock;
			}
		}
	}

	void Field2D::deactivate_lock(waveX::LockManager<Godzilla::Field2D, Godzilla::vecxd> *lock) {
		if (_lock && (lock == _lockptr)) {
			lock->_ptr = nullptr;
			_lock = false;
			_lockptr = nullptr;
		}
	}

	bool Field2D::is_equal(const Godzilla::Field2D &field2D, const bool &name_except) const {
		if (name_except) {
			return (_geom2D.is_equal(field2D.get_geom2D(), false) && (_data == field2D.get_cdata())) ? true : false;
		}
		else {
			return (_geom2D.is_equal(field2D.get_geom2D(), true) && (_data == field2D.get_cdata()) && (_name == field2D.get_name())) ? true : false;
		}
	}

}