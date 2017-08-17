#pragma once
#include "geometry1D.h"
#include "lock_manager.h"

namespace Godzilla {
	class Velocity1D {
		public:
			// Constructors
			Velocity1D();
			Velocity1D(const Godzilla::Geometry1D &geom1D, const std::string &name = "");
			Velocity1D(const Godzilla::Geometry1D &geom1D, const double &scalar, const std::string &name = "");
			Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::xd &scalar, const std::string &name = "");
			Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecd &data, const std::string &name = "");
			Velocity1D(const Godzilla::Geometry1D &geom1D, const Godzilla::vecxd &data, const std::string &name = "");
			Velocity1D(const Godzilla::Velocity1D &vel1D);
			Velocity1D(Godzilla::Velocity1D &&vel1D);

			// Public methods
			Godzilla::Velocity1D& operator=(const Godzilla::Velocity1D &vel1D);
			Godzilla::Velocity1D& operator=(Godzilla::Velocity1D &&vel1D);
			Godzilla::Geometry1D get_geom1D() const { return _geom1D; }
			const Godzilla::vecxd& get_cdata() const { return _data; }
			std::string get_name() const { return _name; }
			size_t get_nelem() const { return _geom1D.get_nX(); }

			void activate_lock(waveX::LockManager<Godzilla::Velocity1D, Godzilla::vecxd> *lock);
			void deactivate_lock(waveX::LockManager<Godzilla::Velocity1D, Godzilla::vecxd> *lock);

			bool is_locked() const { return _lock; }
			bool is_data_equal(const Godzilla::vecxd &data) const { return _data == data; }
			bool is_data_equal(const Godzilla::Velocity1D &vel1D) const { return _data == vel1D.get_cdata(); }

		private:
			// Private members
			Godzilla::Geometry1D _geom1D;   // id = 0
			Godzilla::vecxd _data;          // id = 1
			std::string _name;              // id = 2
			bool _lock;                     // id = 3
			void *_lockptr;                 // id = 4
	};
}