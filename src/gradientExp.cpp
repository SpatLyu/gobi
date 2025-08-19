#include <Rcpp.h>
#include "gradient.h"

// [[Rcpp::export]]
Rcpp::NumericVector gradient_scalar(Rcpp::Function f, Rcpp::NumericVector x0, double h = 0) {
  std::vector<double> x(x0.begin(), x0.end());

  auto f_cpp = [&](const std::vector<double>& xx) {
    Rcpp::NumericVector res = f(Rcpp::wrap(xx));
    return (double)res[0];
  };

  if (h == 0) h = std::cbrt(std::numeric_limits<double>::epsilon());
  std::vector<double> g = gradient(f_cpp, x, h);

  return Rcpp::wrap(g);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix jacobian_vector(Rcpp::Function F, Rcpp::NumericVector x0, double h = 0) {
  std::vector<double> x(x0.begin(), x0.end());

  auto F_cpp = [&](const std::vector<double>& xx) {
    Rcpp::NumericVector res = F(Rcpp::wrap(xx));
    return std::vector<double>(res.begin(), res.end());
  };

  if (h == 0) h = std::cbrt(std::numeric_limits<double>::epsilon());
  std::vector<std::vector<double>> J = gradient(F_cpp, x, h);

  Rcpp::NumericMatrix out(J.size(), J[0].size());
  for (size_t i = 0; i < J.size(); i++) {
    for (size_t j = 0; j < J[i].size(); j++) {
      out(i,j) = J[i][j];
    }
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector gradient_1d(Rcpp::NumericVector F, double h = 1.0) {
  std::vector<double> f(F.begin(), F.end());
  std::vector<double> g = gradient(f, h);
  return Rcpp::wrap(g);
}

// [[Rcpp::export]]
Rcpp::List gradient_2d(Rcpp::NumericMatrix F, double hx = 1.0, double hy = 1.0) {
  size_t n = F.nrow(), m = F.ncol();
  std::vector<std::vector<double>> f(n, std::vector<double>(m));
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < m; j++)
      f[i][j] = F(i,j);

  Gradient2D G = gradient(f, hx, hy);

  Rcpp::NumericMatrix dX(n, m), dY(n, m);
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < m; j++) {
      dX(i,j) = G.dX[i][j];
      dY(i,j) = G.dY[i][j];
    }

    return Rcpp::List::create(
      Rcpp::Named("X") = dX,
      Rcpp::Named("Y") = dY
    );
}

