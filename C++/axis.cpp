#include "axis.h"
#include <iostream>
#include <cassert>

namespace waveX {
	// Constructors
	Axis::Axis(const std::vector<size_t> &n) : _n(0,0), _o(0,0.), _d(0,1.), _label(0,""), _ndims(0) {
		const size_t ndims = n.size();
		bool flag = ((ndims > 0) ? true : false);
		for (auto &i : n) {
			if (i == 0) {
				flag = false;
				break;
			}
		}
		if (flag) {
			_n = n;
			_o.assign(ndims, 0.);
			_d.assign(ndims, 1.);
			_label.assign(ndims, "");
			_ndims = ndims;
		}
	}

	Axis::Axis(const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d) 
		: _n(0, 0), _o(0, 0.), _d(0, 1.), _label(0, ""), _ndims(0) {

		const size_t ndims = n.size();
		bool flag = ((ndims > 0) ? true : false);
		for (auto &i : n) {
			if (i == 0) {
				flag = false;
				break;
			}
		}
		if ((ndims != o.size()) || (ndims != d.size())) flag = false;

		if (flag) {
			_n = n;
			_o = o;
			_d = d;
			_label.assign(ndims, "");
			_ndims = ndims;
		}
	}

	Axis::Axis(const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d, const std::vector<std::string> &label) 
		: _n(0, 0), _o(0, 0.), _d(0, 1.), _label(0, ""), _ndims(0) {

		const size_t ndims = n.size();
		bool flag = ((ndims > 0) ? true : false);
		for (auto &i : n) {
			if (i == 0) {
				flag = false;
				break;
			}
		}
		if ((ndims != o.size()) || (ndims != d.size()) || (ndims != label.size())) flag = false;

		if (flag) {
			_n = n;
			_o = o;
			_d = d;
			_label = label;
			_ndims = ndims;
		}
	}

	Axis::Axis(waveX::Axis &&axis) {
		_n = std::move(axis._n);
		_o = std::move(axis._o);
		_d = std::move(axis._d);
		_label = std::move(axis._label);
		_ndims = axis.get_ndims();
		axis._ndims = 0;
	}

	// Public methods
	waveX::Axis& Axis::operator=(const waveX::Axis &axis) {
		_n = axis.get_n();
		_o = axis.get_o();
		_d = axis.get_d();
		_label = axis.get_label();
		_ndims = axis.get_ndims();
		return *this;
	}

	waveX::Axis& Axis::operator=(waveX::Axis &&axis) {
		_n = std::move(axis._n);
		_o = std::move(axis._o);
		_d = std::move(axis._d);
		_label = std::move(axis._label);
		_ndims = axis.get_ndims();
		axis._ndims = 0;
		return *this;
	}

	size_t Axis::get_nelem() const {
		if (this->is_empty()) {
			return 0;
		}
		else {
			size_t nelem = 1;
			for (size_t i = 0; i < _ndims; ++i) {
				nelem *= _n[i];
			}
			return nelem;
		}
	}

	size_t Axis::get_n(const size_t &dim) const {
		if (dim < _ndims) {
			return _n[dim];
		}
		else {
			std::cerr << "dim exceeds _ndims" << std::endl;
			assert(1 == 2);
		}
	}

	double Axis::get_o(const size_t &dim) const {
		if (dim < _ndims) {
			return _o[dim];
		}
		else {
			std::cerr << "dim exceeds _ndims" << std::endl;
			assert(1 == 2);
		}
	}

	double Axis::get_d(const size_t &dim) const {
		if (dim < _ndims) {
			return _d[dim];
		}
		else {
			std::cerr << "dim exceeds _ndims" << std::endl;
			assert(1 == 2);
		}
	}

	std::string Axis::get_label(const size_t &dim) const {
		if (dim < _ndims) {
			return _label[dim];
		}
		else {
			std::cerr << "dim exceeds _ndims" << std::endl;
			assert(1 == 2);
		}
	}

	void Axis::shift_o(const double &shift) {
		for (auto &i : _o) {
			i += shift;
		}
	}

	void Axis::shift_o(const std::vector<double> &shift) {
		if (shift.size() != _ndims) {
			return;
		}
		else {
			for (size_t i = 0; i < _ndims; ++i) {
				_o[i] += shift[i];
			}
		}
	}

	void Axis::shift_d(const double &shift) {
		for (auto &i : _d) {
			i += shift;
		}
	}

	void Axis::shift_d(const std::vector<double> &shift) {
		if (shift.size() != _ndims) {
			return;
		}
		else {
			for (size_t i = 0; i < _ndims; ++i) {
				_d[i] += shift[i];
			}
		}
	}

	void Axis::scale_o(const double &scale) {
		for (auto &i : _o) {
			i *= scale;
		}
	}

	void Axis::scale_o(const std::vector<double> &scale) {
		if (scale.size() != _ndims) {
			return;
		}
		else {
			for (size_t i = 0; i < _ndims; ++i) {
				_o[i] *= scale[i];
			}
		}
	}

	void Axis::scale_d(const double &scale) {
		for (auto &i : _d) {
			i *= scale;
		}
	}

	void Axis::scale_d(const std::vector<double> &scale) {
		if (scale.size() != _ndims) {
			return;
		}
		else {
			for (size_t i = 0; i < _ndims; ++i) {
				_d[i] *= scale[i];
			}
		}
	}

	void Axis::set_o(const std::vector<double> &o) {
		if (o.size() != _ndims) {
			return;
		}
		else {
			_o = o;
		}
	}

	void Axis::set_d(const std::vector<double> &d) {
		if (d.size() != _ndims) {
			return;
		}
		else {
			_d = d;
		}
	}

	void Axis::set_axis(const std::vector<size_t> &n) {
		const size_t ndims = n.size();
		bool flag = ((ndims > 0) ? true : false);
		for (auto &i : n) {
			if (i == 0) {
				flag = false;
				break;
			}
		}
		if (flag) {
			_n = n;
			_o.assign(ndims, 0.);
			_d.assign(ndims, 1.);
			_label.assign(ndims, "");
			_ndims = ndims;
		}
		else {
			_n.clear();
			_o.clear();
			_d.clear();
			_label.clear();
			_ndims = 0;
		}
	}

	void Axis::set_axis(const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d) {
		const size_t ndims = n.size();
		bool flag = ((ndims > 0) ? true : false);
		for (auto &i : n) {
			if (i == 0) {
				flag = false;
				break;
			}
		}
		if ((ndims != o.size()) || (ndims != d.size())) flag = false;
		if (flag) {
			_n = n;
			_o = o;
			_d = d;
			_label.assign(ndims, "");
			_ndims = ndims;
		}
		else {
			_n.clear();
			_o.clear();
			_d.clear();
			_label.clear();
			_ndims = 0;
		}
	}

	void Axis::set_axis(const std::vector<size_t> &n, const std::vector<double> &o, const std::vector<double> &d, const std::vector<std::string> &label) {
		const size_t ndims = n.size();
		bool flag = ((ndims > 0) ? true : false);
		for (auto &i : n) {
			if (i == 0) {
				flag = false;
				break;
			}
		}
		if ((ndims != o.size()) || (ndims != d.size()) || (ndims != label.size())) flag = false;
		if (flag) {
			_n = n;
			_o = o;
			_d = d;
			_label = label;
			_ndims = ndims;
		}
		else {
			_n.clear();
			_o.clear();
			_d.clear();
			_label.clear();
			_ndims = 0;
		}
	}

	bool Axis::has_same_shape(const waveX::Axis &axis) const {
		return ((_n == axis.get_n()) ? true : false);
	}

	bool Axis::is_equal(const waveX::Axis &axis) const {
		if ((_n != axis.get_n()) || (_o != axis.get_o()) || (_d != axis.get_d()) || (_label != axis.get_label())) {
			return false;
		}
		else {
			return true;
		}
	}

	void Axis::print_n() const {
		std::cout << "Printing \"_n\" of axis :" << std::endl;
		for (size_t i = 0; i < _ndims; ++i) {
			std::cout << "_n[" << i << "] = " << _n[i] << std::endl;
		}
	}

	void Axis::print_o() const {
		std::cout << "Printing \"_o\" of axis :" << std::endl;
		for (size_t i = 0; i < _ndims; ++i) {
			std::cout << "_o[" << i << "] = " << _o[i] << std::endl;
		}
	}

	void Axis::print_d() const {
		std::cout << "Printing \"_d\" of axis :" << std::endl;
		for (size_t i = 0; i < _ndims; ++i) {
			std::cout << "_d[" << i << "] = " << _d[i] << std::endl;
		}
	}

	void Axis::print_label() const {
		std::cout << "Printing \"_label\" of axis :" << std::endl;
		for (size_t i = 0; i < _ndims; ++i) {
			std::cout << "_label[" << i << "] = " << _label[i] << std::endl;
		}
	}

	void Axis::print_ndims() const {
		std::cout << "Printing \"_ndims\" of axis :" << std::endl;
		std::cout << _ndims << std::endl;
	}
}
