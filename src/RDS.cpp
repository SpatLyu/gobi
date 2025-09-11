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
                   const std::vector<double>& time_interval) {
  const size_t n = t.size();
  const size_t num_types = 2;  // For 1D, we have 2 types (X_d >=0 and X_d <0)

  RDSResult result;
  result.score_list.resize(n, std::vector<std::vector<double>>(n, std::vector<double>(num_types, 0.0)));
  result.t_1.resize(n, std::vector<double>(n, 0.0));
  result.t_2.resize(n, std::vector<double>(n, 0.0));
  result.num_types = num_types;

  std::vector<double> f = gradient(Y, time_interval);

  for (size_t t_ori = 0; t_ori < n; ++t_ori) {
    for (size_t t_prime = 0; t_prime < n; ++t_prime) {
      result.t_1[t_ori][t_prime] = t[t_ori];
      result.t_2[t_ori][t_prime] = t[t_prime];

      double X_d = X[t_ori] - X[t_prime];
      double f_d = f[t_ori] - f[t_prime];
      double score_tmp = X_d * f_d;

      if (X_d >= 0) {  // Type 1
        result.score_list[t_ori][t_prime][0] = score_tmp;
      } else {          // Type 2
        result.score_list[t_ori][t_prime][1] = -score_tmp;
      }
    }
  }

  return result;
}

/**
 * @brief Helper function to compute the type index based on difference signs
 *
 * @param diffs Vector of differences (X_d, Y_d, etc.)
 * @return size_t Type index (0 to 2^N-1)
 */
size_t compute_type_index(const std::vector<double>& diffs) {
  size_t index = 0;
  for (size_t i = 0; i < diffs.size(); ++i) {
    if (diffs[i] >= 0) {
      index += (1 << (diffs.size() - 1 - i));
    }
  }
  return index;
}

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
                   bool is_ns = false, bool is_ps = false) {
  // Validate inputs
  if (data.empty()) {
    throw std::invalid_argument("Data vector cannot be empty");
  }

  const size_t n = t.size();
  const size_t num_dimensions = data.size();
  const size_t num_types = 1 << num_dimensions;  // 2^N types for N dimensions

  if (is_ns && is_ps) {
    throw std::invalid_argument("Cannot be both ns and ps version");
  }

  RDSResult result;
  result.score_list.resize(n, std::vector<std::vector<double>>(n, std::vector<double>(num_types, 0.0)));
  result.t_1.resize(n, std::vector<double>(n, 0.0));
  result.t_2.resize(n, std::vector<double>(n, 0.0));
  result.num_types = num_types;

  std::vector<double> f = gradient(K, time_interval);

  for (size_t t_ori = 0; t_ori < n; ++t_ori) {
    for (size_t t_prime = 0; t_prime < n; ++t_prime) {
      result.t_1[t_ori][t_prime] = t[t_ori];
      result.t_2[t_ori][t_prime] = t[t_prime];

      // Compute all differences
      std::vector<double> diffs(num_dimensions);
      for (size_t dim = 0; dim < num_dimensions; ++dim) {
        diffs[dim] = data[dim][t_ori] - data[dim][t_prime];
      }

      double K_d = K[t_ori] - K[t_prime];
      double f_d = f[t_ori] - f[t_prime];

      // Compute the base score (product of all differences * f_d)
      double score_tmp = f_d;
      for (double diff : diffs) {
        score_tmp *= diff;
      }

      // For ns/ps versions, multiply by K_d and check condition
      if (is_ns || is_ps) {
        score_tmp *= K_d;

        // Skip if condition not met
        if ((is_ns && K_d >= 0) || (is_ps && K_d < 0)) {
          continue;
        }
      }

      // Compute type index based on signs of differences
      size_t type_index = compute_type_index(diffs);

      // Determine sign based on number of negative differences
      size_t num_neg = std::count_if(diffs.begin(), diffs.end(), [](double x) { return x < 0; });
      double sign = (num_neg % 2 == 0) ? 1.0 : -1.0;

      result.score_list[t_ori][t_prime][type_index] = sign * score_tmp;
    }
  }

  return result;
}

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
                           const std::vector<double>& time_interval) {
  return RDS_dimN(data, K, t, time_interval, false, false);
}

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
                      const std::vector<double>& time_interval) {
  return RDS_dimN(data, K, t, time_interval, true, false);
}

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
                      const std::vector<double>& time_interval) {
  return RDS_dimN(data, K, t, time_interval, false, true);
}
