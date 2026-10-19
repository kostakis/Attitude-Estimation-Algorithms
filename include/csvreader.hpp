#ifndef CSVREADER_HPP
#define CSVREADER_HPP

#include "pch.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using std::vector;

class CsvReader {
  private:
    std::string fileName;
    char delimiter;

    vector<vector<std::string>> data;
    MatrixXd eigenData;

    int rows{0};
    int cols{0};

    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if(first == std::string::npos)
            return ""; // No content

        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

  public:
    CsvReader(const std::string& fileName, char delimiter = ',') : fileName(fileName), delimiter(delimiter) {}

    void read() {
        data.clear();
        std::ifstream file(fileName);

        if(!file.is_open()) {
            throw std::runtime_error("Could not open file: " + fileName +
                "Binaries should be run from the root folder");
        }

        std::string line;
        while(std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string value;

            while(std::getline(ss, value, delimiter)) {
                value = trim(value);
                // std::cout << "Value read: [" << value << "]" << std::endl;
                row.push_back(value);
            }

            if(!row.empty()) { // Only add non-empty rows
                data.push_back(row);
                rows++;
                if(cols == 0) { // No need to re-check
                    cols = static_cast<int>(row.size());
                }
            }
        }

        file.close();

        // Now allocate and fill Eigen matrix
        eigenData = MatrixXd(rows, cols);

        for(int i = 0; i < rows; ++i) {
            if(data[i].size() != cols) {
                throw std::runtime_error("Inconsistent number of columns in row " + std::to_string(i));
            }
            for(int j = 0; j < cols; ++j) {
                eigenData(i, j) = std::stod(data[i][j]);
            }
        }
    }

    void printData() const {
        if(data.empty()) {
            std::cout << "No data to print." << std::endl;
            return;
        }

        for(size_t i = 0; i < data.size(); ++i) {
            std::cout << "Row " << i + 1 << ": ";
            for(size_t j = 0; j < data[i].size(); ++j) {
                std::cout << "[" << data[i][j] << "] ";
            }
            std::cout << std::endl;
        }
    }

    void printStats() const {
        printf("%s rows: %d Cols: %d (%dx%d)\n", fileName.c_str(), rows, cols, rows, cols);
    }

    void peek() const {
        for(int i = 0; i < 10; i++) {
            std::cout << eigenData.row(i) << std::endl;
        }
    }

    const vector<vector<std::string>>& getData() const {
        return data;
    }

    MatrixXd getEigenData() const { // Give a copy
        return eigenData;
    }

    MatrixXd& getEigenData() { // Give a reference
        return eigenData;
    }

    MatrixXd getColumn(int col) {
        assert(col < cols);
        return eigenData.col(col);
    }

    MatrixXd getManyCols(int col1, int col2) {
        assert(eigenData.cols() < col1);
        assert(eigenData.cols() < col2);

        auto matrix1 = eigenData.col(col1);
        auto matrix2 = eigenData.col(col2);

        assert(matrix1.cols() + matrix2.cols() == 2);

        MatrixXd newMatrix(matrix1.rows(), 2);
        newMatrix << matrix1, matrix2;

        return newMatrix;
    }

    MatrixXd getManyCols(int col1, int col2, int col3) {
        assert(eigenData.cols() < col1);
        assert(eigenData.cols() < col2);
        assert(eigenData.cols() < col3);

        auto matrix1 = eigenData.col(col1);
        auto matrix2 = eigenData.col(col2);
        auto matrix3 = eigenData.col(col3);

        assert(matrix1.cols() + matrix2.cols() + matrix3.cols() == 3);

        MatrixXd newMatrix(matrix1.rows(), 3);
        newMatrix << matrix1, matrix2, matrix3;

        return newMatrix;
    }

    unsigned long getRows() const {
        return rows;
    }

    unsigned long getCols() const {
        return cols;
    }
};

#endif // CSVREADER_HPP
