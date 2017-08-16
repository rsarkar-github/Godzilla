#pragma once
#include "common.h"
#include <string>

namespace waveX {
	class Axis {
		public:
			// Constructors
			Axis() : _n(0, 0), _o(0, 0.), _d(0, 1.), _label(0, ""), _ndims(0) {}
			Axis(const std::vector<size_t> &n);
			Axis(const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d);
			Axis(const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d, const std::vector<std::string> &label);
			Axis(const waveX::Axis &axis) : _n(axis.get_n()), _o(axis.get_o()), _d(axis.get_d()), _label(axis.get_label()), _ndims(axis.get_ndims()) {}
			Axis(waveX::Axis &&axis);

			// Public methods
			waveX::Axis& operator=(const waveX::Axis &axis);
			waveX::Axis& operator=(waveX::Axis &&axis);
			const std::vector<size_t>& get_n() const { return _n; }
			const std::vector<double>& get_o() const { return _o; }
			const std::vector<double>& get_d() const { return _d; }
			const std::vector<std::string>& get_label() const { return _label; }
			size_t get_ndims() const { return _ndims; }
			size_t get_nelem() const;
			size_t get_n(const size_t &dim) const;
			double get_o(const size_t &dim) const;
			double get_d(const size_t &dim) const;
			std::string get_label(const size_t &dim) const;

			void shift_o(const double &shift);
			void shift_o(const std::vector<double> &shift);
			void shift_d(const double &shift);
			void shift_d(const std::vector<double> &shift);
			void scale_o(const double &scale);
			void scale_o(const std::vector<double> &scale);
			void scale_d(const double &scale);
			void scale_d(const std::vector<double> &scale);
			void set_o(const std::vector<double> &o);
			void set_d(const std::vector<double> &d);
			void set_axis(const std::vector<size_t> &n);
			void set_axis (const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d);
			void set_axis(const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d, const std::vector<std::string> &label);
			
			bool is_empty() const { return !_ndims; }
			bool has_same_shape(const waveX::Axis &axis) const;
			bool is_equal(const waveX::Axis &axis) const;

			void print_n() const;
			void print_o() const;
			void print_d() const;
			void print_label() const;
			void print_ndims() const;

		private:
			// Private members
			std::vector<size_t> _n;             // id = 0
			std::vector<double> _o;				// id = 1
			std::vector<double> _d;				// id = 2
			std::vector<std::string> _label;	// id = 3
			size_t _ndims;						// id = 4
	};
}