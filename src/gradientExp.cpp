#include <Rcpp.h>
#include "gradient.h"

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector gradient_cont1d(Rcpp::Function f,
                                    Rcpp::NumericVector x0,
                                    double h = 0) {
  std::vector<double> x(x0.begin(), x0.end());

  auto f_cpp = [&](const std::vector<double>& xx) {
    Rcpp::NumericVector res = f(Rcpp::wrap(xx));
    return (double)res[0];
  };

  if (h == 0) h = std::cbrt(std::numeric_limits<double>::epsilon());
  std::vector<double> g = gradient(f_cpp, x, h);

  return Rcpp::wrap(g);
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericMatrix gradient_cont2d(Rcpp::Function F,
                                    Rcpp::NumericVector x0,
                                    double h = 0) {
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

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector gradient_disc1d(Rcpp::NumericVector F,
                                    Rcpp::NumericVector h = Rcpp::NumericVector::create(1.0)) {
  std::vector<double> f(F.begin(), F.end());
  std::vector<double> h_std(h.begin(), h.end());
  std::vector<double> g = gradient(f, h_std);
  return Rcpp::wrap(g);
}

// [[Rcpp::export(rng = false)]]
Rcpp::List gradient_disc2d(Rcpp::NumericMatrix F,
                           Rcpp::NumericVector hx = Rcpp::NumericVector::create(1.0),
                           Rcpp::NumericVector hy = Rcpp::NumericVector::create(1.0)) {
  size_t n = F.nrow(), m = F.ncol();
  std::vector<std::vector<double>> f(n, std::vector<double>(m));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      f[i][j] = F(i,j);
    }
  }
  std::vector<double> hx_std(hx.begin(), hx.end());
  std::vector<double> hy_std(hy.begin(), hy.end());

  Gradient2D G = gradient(f, hx_std, hy_std);

  Rcpp::NumericMatrix dX(n, m), dY(n, m);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      dX(i,j) = G.dX[i][j];
      dY(i,j) = G.dY[i][j];
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("X") = dX,
    Rcpp::Named("Y") = dY
  );
}
