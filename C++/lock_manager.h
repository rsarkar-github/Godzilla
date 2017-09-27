#pragma once
#include <iostream>

namespace waveX {
	template <typename T, typename P>
	class LockManager {
		public:
			// Constructors
			LockManager() = delete;

			LockManager(T& object, const int &id) : _id(id) {
				if (object.is_locked()) {
					_ptr = nullptr;
				}
				else {
					object.activate_lock(this);
				}
			}
			
			// Public methods
			int get_id() const { return _id; }

			bool is_active() const {
				return !(_ptr == nullptr);
			}

			void activate_lock(T& object) {
				if (_ptr == nullptr) {
					if (!object.is_locked()) {
						object.activate_lock(this);
					}
				}
				else {
					std::cerr << "Lock is activated. Deactivate before activating." << std::endl;
				}
			}

			void deactivate_lock(T& object) {
				if (!(_ptr == nullptr)) {
					object.deactivate_lock(this);
				}
				else {
					std::cerr << "Lock is deactivated. Activate before deactivating." << std::endl;
				}
			}

		    // Public members
			P *_ptr;

		private:
			// Private members
			int _id;
	};
}