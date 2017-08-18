#include "field1D.h"
#include <cmath>

namespace Godzilla {
	// Constructors
	Field1D::Field1D() : _geom1D(Godzilla::Geometry1D()), _name(""), _lock(false), _lockptr(nullptr) {
		_data.assign(_geom1D.get_nX(), 0.);
		_data.shrink_to_fit();
	}

	Field1D::Field1D(const Godzilla::Geometry1D &geom1D, const std::string &name) : _geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {
		_data.assign(geom1D.get_nX(), 0.);
		_data.shrink_to_fit();
	}

	Field1D::Field1D(const Godzilla::Geometry1D &geom1D, const double &scalar, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {

		_data.assign(geom1D.get_nX(), scalar);
		_data.shrink_to_fit();
	}

	Field1D::Field1D(const Godzilla::Geometry1D &geom1D, const Godzilla::xd &scalar, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {

		_data.assign(geom1D.get_nX(), scalar);
		_data.shrink_to_fit();
	}

	Field1D::Field1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecd &data, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom1D.get_nX();
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

	Field1D::Field1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecxd &data, const std::string &name) :
		_geom1D(geom1D), _name(name), _lock(false), _lockptr(nullptr) {

		size_t n = geom1D.get_nX();
		if (data.size() == n) {
			_data = data;
		}
		else {
			_data.assign(n, 0.);
		}
		_data.shrink_to_fit();
	}

	Field1D::Field1D(const Godzilla::Field1D &field1D) : _lock(false), _lockptr(nullptr) {
		_geom1D = field1D.get_geom1D();
		_data = field1D.get_cdata();
		_name = field1D.get_name();
		_data.shrink_to_fit();
	}

	Field1D::Field1D(Godzilla::Field1D &&field1D) : _lock(false), _lockptr(nullptr) {
		_geom1D = field1D.get_geom1D();
		_data = std::move(field1D._data);
		_name = std::move(field1D._name);
	}

	// Public methods
	Godzilla::Field1D& Field1D::operator=(const Godzilla::Field1D &field1D) {
		if (!_lock) {
			_geom1D = field1D.get_geom1D();
			_data = field1D.get_cdata();
			_name = field1D.get_name();
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Field1D locked. Field1D unchanged." << std::endl;
			return *this;
		}
		_data.shrink_to_fit();
	}

	Godzilla::Field1D& Field1D::operator=(Godzilla::Field1D &&field1D) {
		if (!_lock) {
			_geom1D = field1D.get_geom1D();
			_data = std::move(field1D._data);
			_name = std::move(field1D._name);
			_lock = false;
			_lockptr = nullptr;
			return *this;
		}
		else {
			std::cerr << "Field1D locked. Field1D unchanged." << std::endl;
			return *this;
		}
	}

	void Field1D::set_data(const Godzilla::vecd &data) {
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
				std::cerr << "Data size does not match or data contains values too big or too small. Field1D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Field1D locked. Field1D unchanged." << std::endl;
		}
	}

	void Field1D::set_data(const Godzilla::vecxd &data) {
		if (!_lock) {
			size_t n = data.size();
			if (this->get_nelem() == n) {
				_data = data;
			}
			else {
				std::cerr << "Data size does not match or data contains values too big or too small. Field1D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Field1D locked. Field1D unchanged." << std::endl;
		}
	}

	void Field1D::setmove_data(Godzilla::vecxd &data) {
		if (!_lock) {
			size_t n = data.size();
			if (this->get_nelem() == n) {
				_data = std::move(data);
			}
			else {
				std::cerr << "Data size does not match or data contains values too big or too small. Field1D unchanged." << std::endl;
			}
		}
		else {
			std::cerr << "Field1D locked. Field1D unchanged." << std::endl;
		}
	}

	void Field1D::set_name(const std::string &name) {
		if (!_lock) {
			_name = name;
		}
		else {
			std::cerr << "Field1D locked. Field1D unchanged." << std::endl;
		}
	}

	void Field1D::activate_lock(waveX::LockManager<Godzilla::Field1D, Godzilla::vecxd> *lock) {
		if (!_lock) {
			if (lock->get_id() == Godzilla::FIELD1D_DATA_ID) {
				lock->_ptr = &_data;
				_lock = true;
				_lockptr = lock;
			}
		}
	}

	void Field1D::deactivate_lock(waveX::LockManager<Godzilla::Field1D, Godzilla::vecxd> *lock) {
		if (_lock && (lock == _lockptr)) {
			lock->_ptr = nullptr;
			_lock = false;
			_lockptr = nullptr;
		}
	}

	bool Field1D::is_equal(const Godzilla::Field1D &field1D, const bool &name_except) {
		if (name_except) {
			return (_geom1D.is_equal(field1D.get_geom1D(), false) && (_data == field1D.get_cdata())) ? true : false;
		}
		else {
			return (_geom1D.is_equal(field1D.get_geom1D(), true) && (_data == field1D.get_cdata()) && (_name == field1D.get_name())) ? true : false;
		}
	}

}