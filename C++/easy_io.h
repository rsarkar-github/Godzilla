#pragma once
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>

namespace wavemod2d {
	class EasyIO {
		public:
			// Constructors
			EasyIO() : _input_format("sep"), _output_format("sep"){}
			EasyIO(std::string input_format, std::string output_format) : _input_format(input_format), _output_format(output_format){}
			EasyIO(const wavemod2d::EasyIO &io_object) : _input_format(io_object.get_input_format()),
				_output_format(io_object.get_output_format()){}

			// Public methods
			std::string get_input_format() const { return _input_format; }
			std::string get_output_format() const { return _output_format; }
			void set_input_format(std::string input_format) { _input_format = input_format; }
			void set_output_format(std::string output_format) { _output_format = output_format; }
			void assign(const wavemod2d::EasyIO &io_object);

			std::vector<float> read_binary(std::string file_path);
			std::vector<float> read_binary(std::string file_path, size_t n1);
			std::vector<std::vector<float> > read_binary(std::string file_path, size_t n1, size_t n2);
			std::vector<std::vector<std::vector<float> > > read_binary(std::string file_path, size_t n1, size_t n2, size_t n3);

			void write_binary(std::string file_path, std::vector<float> &vec);
			void write_binary(std::string file_path, std::vector<double> &vec);
			void write_binary(std::string file_path, std::vector<std::vector<float> > &vec);
			void write_binary(std::string file_path, std::vector<std::vector<std::vector<float> > > &vec);

			void write_header(std::string file_path, std::vector<std::string> &labels, std::vector<std::string> &values);
			void append_header(std::string file_path, std::vector<std::string> &labels, std::vector<std::string> &values);
			void append_param(std::string file_path, std::string label, std::string value);
			std::string read_param(std::string file_path, std::string label);

		private:
			// Private members
			std::string _input_format;
			std::string _output_format;

			// Private methods
	};
}