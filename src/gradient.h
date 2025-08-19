/*
 * gradient.h
 *
 * Declarations for numerical differentiation utilities.
 * Provides gradient and Jacobian operators, as well as
 * discrete 1D/2D grid gradients.
 *
 * Author: Wenbo Lv
 * Date:   2025-08-20
 */

#ifndef GRADIENT_H
#define GRADIENT_H

#include <vector>
#include <functional>
#include <limits>
#include <cmath>

// ========== 2D Gradient structure ==========
struct Gradient2D {
  std::vector<std::vector<double>> dX;
  std::vector<std::vector<double>> dY;
};

// ========== Function prototypes ==========

// Case 1: Gradient of scalar function f: R^n -> R
std::vector<double> gradient(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0,
    double h = std::cbrt(std::numeric_limits<double>::epsilon())
);

// Case 2: Jacobian of vector function F: R^n -> R^m
std::vector<std::vector<double>> gradient(
    const std::function<std::vector<double>(const std::vector<double>&)>& F,
    const std::vector<double>& x0,
    double h = std::cbrt(std::numeric_limits<double>::epsilon())
);

// Case 3a: Gradient of 1D data
// h can be a scalar (uniform spacing) or a vector of coordinates.
std::vector<double> gradient(
    const std::vector<double>& F,
    const std::vector<double>& h
);

// Case 3b: Gradient of 2D data
// hx, hy can be scalars (uniform spacing) or coordinate vectors.
Gradient2D gradient(
    const std::vector<std::vector<double>>& F,
    const std::vector<double>& hx,
    const std::vector<double>& hy
);

#endif // GRADIENT_H
