#ifndef RDS_H
#define RDS_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include "gradient.h"

/**
 * @brief Structure to hold RDS function results
 *
 * Contains the score matrix, time point matrices, and the number of types
 */
struct RDSResult {
  std::vector<std::vector<std::vector<double>>> score_list;  // 3D score matrix [t1][t2][type]
  std::vector<std::vector<double>> t_1;                      // Time point matrix 1
  std::vector<std::vector<double>> t_2;                      // Time point matrix 2
  size_t num_types;                                          // Number of types (2^N for N dimensions)
};

/**
 * @brief Computes the RDS score for 1D data (base case)
 *
 * @param X Input data vector
 * @param Y Response vector
 * @param t Time points vector
 * @param time_interval Time interval vector
 * @return RDSResult Result structure with scores and time matrices
 */
RDSResult RDS_dim1(const std::vector<double>& X,
                   const std::vector<double>& Y,
                   const std::vector<double>& t,
                   const std::vector<double>& time_interval);

/**
 * @brief Computes the RDS score for N-dimensional data (general case)
 *
 * @param data Input data vectors (X, Y, Z, etc.)
 * @param K Response vector
 * @param t Time points vector
 * @param time_interval Time interval vector
 * @param is_ns If true, uses negative K_d condition (ns version)
 * @param is_ps If true, uses positive K_d condition (ps version)
 * @return RDSResult Result structure with scores and time matrices
 */
RDSResult RDS_dimN(const std::vector<std::vector<double>>& data,
                   const std::vector<double>& K,
                   const std::vector<double>& t,
                   const std::vector<double>& time_interval,
                   bool is_ns = false, bool is_ps = false);

/**
 * @brief Computes the RDS score for N-dimensional data (normal version)
 *
 * @param data Input data vectors (X, Y, Z, etc.)
 * @param K Response vector
 * @param t Time points vector
 * @param time_interval Time interval vector
 * @return RDSResult Result structure with scores and time matrices
 */
RDSResult RDS_dimN_general(const std::vector<std::vector<double>>& data,
                           const std::vector<double>& K,
                           const std::vector<double>& t,
                           const std::vector<double>& time_interval);

/**
 * @brief Computes the RDS score for N-dimensional data (ns version)
 *
 * @param data Input data vectors (X, Y, Z, etc.)
 * @param K Response vector
 * @param t Time points vector
 * @param time_interval Time interval vector
 * @return RDSResult Result structure with scores and time matrices
 */
RDSResult RDS_dimN_ns(const std::vector<std::vector<double>>& data,
                      const std::vector<double>& K,
                      const std::vector<double>& t,
                      const std::vector<double>& time_interval);

/**
 * @brief Computes the RDS score for N-dimensional data (ps version)
 *
 * @param data Input data vectors (X, Y, Z, etc.)
 * @param K Response vector
 * @param t Time points vector
 * @param time_interval Time interval vector
 * @return RDSResult Result structure with scores and time matrices
 */
RDSResult RDS_dimN_ps(const std::vector<std::vector<double>>& data,
                      const std::vector<double>& K,
                      const std::vector<double>& t,
                      const std::vector<double>& time_interval);

#endif // RDS_H
