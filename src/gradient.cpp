/*
 * gradient.cpp
 *
 * Numerical differentiation utilities implemented in C++.
 * This file provides a unified interface `gradient` supporting:
 *   1. Gradient of a scalar function f: R^n -> R (similar to pracma's grad).
 *   2. Jacobian of a vector function F: R^n -> R^m (similar to pracma's jacobian).
 *   3. Numerical gradient of discrete 1D/2D data arrays (similar to pracma's gradient).
 *
 * Methods use central finite differences for accuracy.
 *
 * Usage examples:
 *   - auto g = gradient(f, x0);                     // scalar f, returns vector
 *   - auto J = gradient(F, x0);                     // vector F, returns matrix
 *   - auto g = gradient(data1D, h);                 // 1D array
 *   - auto g = gradient(data2D, hX, hY);            // 2D array
 *
 * Author: Wenbo Lv
 * Date:   2025-08-20
 */

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>

// ======================= Case 1: Gradient of scalar function =======================
std::vector<double> gradient(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0,
    double h = std::cbrt(std::numeric_limits<double>::epsilon())
) {
  if (x0.empty()) throw std::invalid_argument("x0 must be non-empty.");

  size_t n = x0.size();
  std::vector<double> grad(n);
  std::vector<double> xplus = x0, xminus = x0;

  for (size_t i = 0; i < n; i++) {
    xplus[i] += h;
    xminus[i] -= h;
    grad[i] = (f(xplus) - f(xminus)) / (2*h);
    xplus[i] = x0[i];
    xminus[i] = x0[i];
  }
  return grad;
}

// ======================= Case 2: Jacobian of vector function =======================
std::vector<std::vector<double>> gradient(
    const std::function<std::vector<double>(const std::vector<double>&)>& F,
    const std::vector<double>& x0,
    double h = std::cbrt(std::numeric_limits<double>::epsilon())
) {
  if (x0.empty()) throw std::invalid_argument("x0 must be non-empty.");

  size_t n = x0.size();
  std::vector<double> f0 = F(x0);
  size_t m = f0.size();

  std::vector<std::vector<double>> J(m, std::vector<double>(n));
  std::vector<double> xplus = x0, xminus = x0;

  for (size_t j = 0; j < n; j++) {
    xplus[j] += h;
    xminus[j] -= h;
    std::vector<double> fplus = F(xplus);
    std::vector<double> fminus = F(xminus);
    for (size_t i = 0; i < m; i++) {
      J[i][j] = (fplus[i] - fminus[i]) / (2*h);
    }
    xplus[j] = x0[j];
    xminus[j] = x0[j];
  }
  return J;
}

// ======================= Case 3a: Gradient of 1D data =======================
std::vector<double> gradient(
    const std::vector<double>& F,
    double h = 1.0
) {
  size_t n = F.size();
  if (n == 0) return {};
  if (n == 1) return {0.0};

  std::vector<double> g(n);
  g[0] = (F[1] - F[0]) / h;
  g[n-1] = (F[n-1] - F[n-2]) / h;
  for (size_t i = 1; i < n-1; i++) {
    g[i] = (F[i+1] - F[i-1]) / (2*h);
  }
  return g;
}

// ======================= Case 3b: Gradient of 2D data =======================
struct Gradient2D {
  std::vector<std::vector<double>> dX;
  std::vector<std::vector<double>> dY;
};

Gradient2D gradient(
    const std::vector<std::vector<double>>& F,
    double hx = 1.0,
    double hy = 1.0
) {
  size_t n = F.size();
  if (n == 0) return {};
  size_t m = F[0].size();

  Gradient2D G;
  G.dX.assign(n, std::vector<double>(m, 0.0));
  G.dY.assign(n, std::vector<double>(m, 0.0));

  // d/dx
  for (size_t i = 0; i < n; i++) {
    G.dX[i][0] = (F[i][1] - F[i][0]) / hx;
    G.dX[i][m-1] = (F[i][m-1] - F[i][m-2]) / hx;
    for (size_t j = 1; j < m-1; j++) {
      G.dX[i][j] = (F[i][j+1] - F[i][j-1]) / (2*hx);
    }
  }

  // d/dy
  for (size_t j = 0; j < m; j++) {
    G.dY[0][j] = (F[1][j] - F[0][j]) / hy;
    G.dY[n-1][j] = (F[n-1][j] - F[n-2][j]) / hy;
    for (size_t i = 1; i < n-1; i++) {
      G.dY[i][j] = (F[i+1][j] - F[i-1][j]) / (2*hy);
    }
  }
  return G;
}

