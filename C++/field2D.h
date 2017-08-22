#pragma once
#include "geometry2D.h"
#include "lock_manager.h"

namespace Godzilla {
	class Field2D {
		public:
			// Constructors
			Field2D();
			Field2D(const Godzilla::Geometry2D &geom2D, const std::string &name = "");
			Field2D(const Godzilla::Geometry2D &geom2D, const double &scalar, const std::string &name = "");
			Field2D(const Godzilla::Geometry2D &geom2D, const Godzilla::xd &scalar, const std::string &name = "");
			Field2D(const Godzilla::Geometry2D &geom2D, const Godzilla::vecd &data, const std::string &name = "");
			Field2D(const Godzilla::Geometry2D &geom2D, const Godzilla::vecxd &data, const std::string &name = "");
			Field2D(const Godzilla::Field2D &field2D);
			Field2D(Godzilla::Field2D &&field2D);

			// Public methods
			Godzilla::Field2D& operator=(const Godzilla::Field2D &field2D);
			Godzilla::Field2D& operator=(Godzilla::Field2D &&field2D);
			Godzilla::Geometry2D get_geom2D() const { return _geom2D; }
			const Godzilla::vecxd& get_cdata() const { return _data; }
			std::string get_name() const { return _name; }
			size_t get_nelem() const { return _geom2D.get_nX() * _geom2D.get_nY(); }

			void set_data(const Godzilla::vecd &data);
			void set_data(const Godzilla::vecxd &data);
			void setmove_data(Godzilla::vecxd &data);
			void set_name(const std::string &name);

			void activate_lock(waveX::LockManager<Godzilla::Field2D, Godzilla::vecxd> *lock);
			void deactivate_lock(waveX::LockManager<Godzilla::Field2D, Godzilla::vecxd> *lock);

			bool is_locked() const { return _lock; }
			bool is_equal(const Godzilla::Field2D &field2D, const bool &name_except = true) const;
			bool is_data_equal(const Godzilla::vecxd &data) const { return _data == data; }
			bool is_data_equal(const Godzilla::Field2D &field2D) const { return _data == field2D.get_cdata(); }

		private:
			// Private members
			Godzilla::Geometry2D _geom2D;   // id = 0
			Godzilla::vecxd _data;          // id = 1
			std::string _name;              // id = 2
			bool _lock;                     // id = 3
			void *_lockptr;                 // id = 4
	};
}