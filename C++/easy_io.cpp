#include "easy_io.h"

namespace wavemod2d {

	// Public methods

	// Assign an EasyIO object
	void EasyIO::assign(const wavemod2d::EasyIO &io_object) {
		_input_format = io_object.get_input_format();
		_output_format = io_object.get_output_format();
	}

	// Private methods
	/*
	Reads a binary file into a 1d float vector.
	*/
	std::vector<float> EasyIO::read_binary(std::string file_path) {
		// Initialize vector
		std::vector<float> v;

		// Open file for reading
		std::ifstream file;
		file.open(file_path, std::ios::in | std::ios::binary);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened. Returning empty vector." << std::endl;
			return v;
		}
		// File opened successfully
		else {
			// Get number of floats in file
			file.seekg(0, std::ios::end);
			std::ios::streampos last = file.tellg();
			file.seekg(std::ios::beg);
			size_t n_elem = last / sizeof(float);

			// Check if n_elem = 0
			if (n_elem == 0) {
				std::cout << "Binary file does not even have 1 float. Returning empty vector." << std::endl;
				file.close();
				return v;
			}

			// Allocate vector
			v.assign(n_elem, 0.);

			// Read the file
			file.read(reinterpret_cast<char*>(v.data()), n_elem * sizeof(float));

			// Close file
			file.close();
		}

		// Return v
		return v;
	}

	/*
	Reads a binary file into a 1d float vector of size n1. If elements in the binary file
	are insufficient, then it is padded with zeros
	*/
	std::vector<float> EasyIO::read_binary(std::string file_path, size_t n1) {
		// Allocate 1d vector
		std::vector<float> v(n1, 0.);

		// Open file for reading
		std::ifstream file;
		file.open(file_path, std::ios::in | std::ios::binary);
		
		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened. Returning vector of zeros." << std::endl;
			return v;
		}
		// File opened successfully
		else {
			// Get number of floats in file
			file.seekg(0, std::ios::end);
			std::ios::streampos last = file.tellg();
			file.seekg(std::ios::beg);
			size_t n_elem = last / sizeof(float);

			// Calculate number of elements to read
			n_elem = std::min(n_elem, n1);
			
			// Check if n_elem = 0
			if (n_elem == 0) {
				std::cout << "Binary file does not even have 1 float. Returning vector of zeros." << std::endl;
				file.close();
				return v;
			}

			// Read the file
			file.read(reinterpret_cast<char*>(v.data()), n_elem * sizeof(float));

			// Close file
			file.close();
		}

		// Return v
		return v;
	}

	/*
	Reads a binary file into a 2d float vector of size n1 x n2. If elements in the binary file
	are insufficient, then it is padded with zeros
	*/
	std::vector<std::vector<float> > EasyIO::read_binary(std::string file_path, size_t n1, size_t n2) {
		// Allocate 2d vector
		std::vector<std::vector<float> > v(n1, std::vector<float>(n2, 0.));

		// Open file for reading
		std::ifstream file;
		file.open(file_path, std::ios::in | std::ios::binary);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened. Returning vector of zeros." << std::endl;
			return v;
		}
		// File opened successfully
		else {
			// Get number of floats in file
			file.seekg(0, std::ios::end);
			std::ios::streampos last = file.tellg();
			file.seekg(std::ios::beg);
			size_t n_elem = last / sizeof(float);

			// Calculate number of elements to read
			// Reset n1 : number of elements of first axis which are full
			// Reset n2 : number of elements in last element of first axis
			n_elem = std::min(n_elem, n1 * n2);
			n1 = n_elem / n2;
			n2 = n_elem - n1 * n2;

			// Check if n_elem = 0
			if (n_elem == 0) {
				std::cout << "Binary file does not even have 1 float. Returning vector of zeros." << std::endl;
				file.close();
				return v;
			}

			// Read the file
			for (size_t i = 0; i < n1; i++) {
				file.read(reinterpret_cast<char*>(v[i].data()), v[i].size() * sizeof(float));
			}
			if(n2 > 0) file.read(reinterpret_cast<char*>(v[n1].data()), n2 * sizeof(float));

			// Close file
			file.close();
		}

		// Return v
		return v;
	}

	/*
	Reads a binary file into a 3d float vector of size n1 x n2 x n3. If elements in the binary file
	are insufficient, then it is padded with zeros
	*/
	std::vector<std::vector<std::vector<float> > > EasyIO::read_binary(std::string file_path, size_t n1, size_t n2, size_t n3) {
		// Allocate 2d vector
		std::vector<std::vector<std::vector<float> > > v(n1, std::vector<std::vector<float> >(n2, std::vector<float>(n3, 0.)));

		// Open file for reading
		std::ifstream file;
		file.open(file_path, std::ios::in | std::ios::binary);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened. Returning vector of zeros." << std::endl;
			return v;
		}
		// File opened successfully
		else {
			// Get number of floats in file
			file.seekg(0, std::ios::end);
			std::ios::streampos last = file.tellg();
			file.seekg(std::ios::beg);
			size_t n_elem = last / sizeof(float);

			// Calculate number of elements to read
			// Reset n1 : number of elements of first axis which are full
			// Reset n2 : number of elements in last element of first axis
			// Reset n3 : number of elements in last element of second axis of last element of first axis
			n_elem = std::min(n_elem, n1 * n2 * n3);
			size_t temp = n_elem;
			n1 = temp / (n2 * n3);
			temp = temp - n1 * n2 * n3;
			n2 = temp / n3;
			n3 = temp - n2 * n3;
			
			// Check if n_elem = 0
			if (n_elem == 0) {
				std::cout << "Binary file does not even have 1 float. Returning vector of zeros." << std::endl;
				file.close();
				return v;
			}
			
			// Read the file
			for (size_t i = 0; i < n1; i++) {
				for (auto &j : v[i]) {
					file.read(reinterpret_cast<char*>(j.data()), j.size() * sizeof(float));
				}
			}
			if (n2 > 0 || n3 > 0) {
				for (size_t i = 0; i < n2; i++) {
					file.read(reinterpret_cast<char*>(v[n1][i].data()), v[n1][i].size() * sizeof(float));
				}
				if (n3 > 0) file.read(reinterpret_cast<char*>(v[n1][n2].data()), n3 * sizeof(float));
			}

			// Close file
			file.close();
		}

		// Return v
		return v;
	}

	// Writes a 1d vector to a binary file
	void EasyIO::write_binary(std::string file_path, std::vector<float> &vec) {
		// Check if vec is empty
		if (vec.empty()) {
			std::cout << "Cannot write empty vector." << std::endl;
			return;
		}

		// Open file for writing
		std::ofstream file;
		file.open(file_path, std::ios::out | std::ios::binary);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			// Write the file
			file.write(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(float));
			// Close file
			file.close();
		}
	}

	// Writes a 1d vector to a binary file (double)
	void EasyIO::write_binary(std::string file_path, std::vector<double> &vec) {
		// Check if vec is empty
		if (vec.empty()) {
			std::cout << "Cannot write empty vector." << std::endl;
			return;
		}

		// Open file for writing
		std::ofstream file;
		file.open(file_path, std::ios::out | std::ios::binary);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			// Write the file
			file.write(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(double));
			// Close file
			file.close();
		}
	}

	// Writes a 2d vector to a binary file
	void EasyIO::write_binary(std::string file_path, std::vector<std::vector<float> > &vec) {
		// Check if vec is empty
		if (vec.empty() || vec[0].empty()) {
			std::cout << "Cannot write empty vector." << std::endl;
			return;
		}

		// Check if 2d vector is of uniform size
		size_t s = vec[0].size();
		for (auto &i : vec) {
			if (i.size() != s) {
				std::cout << "Cannot write 2d vector of non-uniform size to file." << std::endl;
				return;
			}
		}

		// Open file for writing
		std::ofstream file;
		file.open(file_path, std::ios::out | std::ios::binary);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			// Write the file
			for (auto &i : vec) {
				file.write(reinterpret_cast<char*>(i.data()), i.size() * sizeof(float));
			}
			// Close file
			file.close();
		}
	}

	// Writes a 3d vector to a binary file
	void EasyIO::write_binary(std::string file_path, std::vector<std::vector<std::vector<float> > > &vec) {
		// Check if vec is empty
		if (vec.empty() || vec[0].empty() || vec[0][0].empty()) {
			std::cout << "Cannot write empty vector." << std::endl;
			return;
		}

		// Check if 3d vector is of uniform size
		size_t s2 = vec[0].size();
		size_t s3 = vec[0][0].size();
		for (auto &i : vec) {
			if (i.size() != s2) {
				std::cout << "Cannot write 2d vector of non-uniform size to file." << std::endl;
				return;
			}
			for (auto &j : i) {
				if (j.size() != s3) {
					std::cout << "Cannot write 2d vector of non-uniform size to file." << std::endl;
					return;
				}
			}
		}

		// Open file for writing
		std::ofstream file;
		file.open(file_path, std::ios::out | std::ios::binary);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			// Write the file
			for (auto &i : vec) {
				for (auto &j : i) {
					file.write(reinterpret_cast<char*>(j.data()), j.size() * sizeof(float));
				}
			}
			// Close file
			file.close();
		}
	}

	// Writes an ascii file whose entries are : labels[i]=values[i]
	void EasyIO::write_header(std::string file_path, std::vector<std::string> &labels, std::vector<std::string> &values) {
		// Check if labels or values is empty, and if they have same size
		if (labels.empty() || values.empty()) {
			std::cout << "Labels and values are empty. Header file not created." << std::endl;
			return;
		}
		if (labels.size() != values.size()) {
			std::cout << "Labels and values must be same size. Header file not created." << std::endl;
			return;
		}

		// Open file for writing
		std::ofstream file;
		file.open(file_path, std::ios::out);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			// Write the file
			for (size_t i = 0; i < labels.size(); i++) {
				file << labels[i] + "=" + values[i] << "\n";
			}
			// Close file
			file.close();
		}
	}
	
	// Appends an ascii file with entries : labels[i]=values[i]
	void EasyIO::append_header(std::string file_path, std::vector<std::string> &labels, std::vector<std::string> &values) {
		// Check if labels or values is empty, and if they have same size
		if (labels.empty() || values.empty()) {
			std::cout << "Labels and values are empty. Header file not created." << std::endl;
			return;
		}
		if (labels.size() != values.size()) {
			std::cout << "Labels and values must be same size. Header file not created." << std::endl;
			return;
		}

		// Open file for appending
		std::ofstream file;
		file.open(file_path, std::ios::out | std::ios::app);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			// Write the file
			for (size_t i = 0; i < labels.size(); i++) {
				file << labels[i] + "=" + values[i] << "\n";
			}
			// Close file
			file.close();
		}
	}

	// Appends an ascii file with the entry : label=value
	void EasyIO::append_param(std::string file_path, std::string label, std::string value) {
		// Check if label = ""
		if (label == "" || value == "") {
			std::cout << "label or value cannot be \"\". File not appended." << std::endl;
			return;
		}

		// Open file for appending
		std::ofstream file;
		file.open(file_path, std::ios::out | std::ios::app);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			// Append the file
			file << label + "=" + value << "\n";
			// Close file
			file.close();
		}
	}

	// Reads value of label from header file
	std::string EasyIO::read_param(std::string file_path, std::string label) {
		// Initialize return type and bool match_found;
		std::string value = "";
		bool match_found = false;

		// Open file for reading
		std::ifstream file;
		file.open(file_path, std::ios::in);

		// Check if file was opened successfully
		if (!file.is_open()) {
			std::cout << "File could not be opened." << std::endl;
		}
		// File opened successfully
		else {
			std::string line = "";
			// Read line
			while (file >> line) {
				// Check if label matches, and if yes, update value
				for (size_t i = 0; i < line.size(); i++) {
					if (line[i] == '=') {
						if (line.substr(0, i) == label) {
							match_found = true;
							value = line.substr(i + 1, line.size() - 2);
						}
						break;
					}
				}
			}
			// Close file
			file.close();
		}
		// Return value
		if (match_found == true) {
			return value;
		}
		else {
			std::cout << "label not found. Returning empty string." << std::endl;
			return value;
		}
	}
}