#pragma once
#include "geometry1D.h"
#include "lock_manager.h"

namespace Godzilla {
	class Field1D {
	public:
		// Constructors
		Field1D();
		Field1D(const Godzilla::Geometry1D &geom1D, const std::string &name = "");
		Field1D(const Godzilla::Geometry1D &geom1D, const double &scalar, const std::string &name = "");
		Field1D(const Godzilla::Geometry1D &geom1D, const Godzilla::xd &scalar, const std::string &name = "");
		Field1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecd &data, const std::string &name = "");
		Field1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecxd &data, const std::string &name = "");
		Field1D(const Godzilla::Field1D &field1D);
		Field1D(Godzilla::Field1D &&field1D);

		// Public methods
		Godzilla::Field1D& operator=(const Godzilla::Field1D &field1D);
		Godzilla::Field1D& operator=(Godzilla::Field1D &&field1D);
		Godzilla::Geometry1D get_geom1D() const { return _geom1D; }
		const Godzilla::vecxd& get_cdata() const { return _data; }
		std::string get_name() const { return _name; }
		size_t get_nelem() const { return _geom1D.get_nX(); }

		void set_data(const Godzilla::vecd &data);
		void set_data(const Godzilla::vecxd &data);
		void setmove_data(Godzilla::vecxd &data);
		void set_name(const std::string &name);

		void activate_lock(waveX::LockManager<Godzilla::Field1D, Godzilla::vecxd> *lock);
		void deactivate_lock(waveX::LockManager<Godzilla::Field1D, Godzilla::vecxd> *lock);

		bool is_locked() const { return _lock; }
		bool is_equal(const Godzilla::Field1D &field1D, const bool &name_except = true);
		bool is_data_equal(const Godzilla::vecxd &data) const { return _data == data; }
		bool is_data_equal(const Godzilla::Field1D &field1D) const { return _data == field1D.get_cdata(); }

	private:
		// Private members
		Godzilla::Geometry1D _geom1D;   // id = 0
		Godzilla::vecxd _data;          // id = 1
		std::string _name;              // id = 2
		bool _lock;                     // id = 3
		void *_lockptr;                 // id = 4
	};
}