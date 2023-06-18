//
// Created by David on 2023/6/17.
//

#include "newton_fractal.h"
#include <iostream>
#include <complex>

namespace nf = newton_fractal;
using std::cout, std::endl;

template <typename float_t, typename complex_t = std::complex<float_t>>
void test_euqation() noexcept {
  nf::point_list<complex_t> points;
  points.emplace_back(1, 2);
  points.emplace_back(3, 4);
  points.emplace_back(5, 6);
  // points.emplace_back(7, 8);
  // points.emplace_back(9, 10);
  // points.emplace_back(11, 12);
  //  points.emplace_back(1, 2);

  nf::newton_equation<complex_t> ne{points};

  fmt::print("equation : {}\n", ne.to_string());

  for (auto& p : points) {
    auto diff = ne.compute_difference(p);

    complex_t norm2;
    newton_fractal::compute_norm2(diff, norm2);

    fmt::print("Difference at ({}+{}i) = {}\n", float_t(p.real()),
               float_t(p.imag()), float_t(norm2.real()));
  }

  complex_t z = 10000;
  for (int i = 0; i < 30; i++) {
    complex_t z_1 = ne.iterate(z);

    fmt::print("({}+{}i) iterates to ({}+{}i)\n", float_t(z.real()),
               float_t(z.imag()), float_t(z_1.real()), float_t(z_1.imag()));
    z = z_1;
  }
}

int main(int argc, char** argv) {
  test_euqation<float>();
  test_euqation<double>();

  // return 0;

  test_euqation<double, fu::complex_type_of<fu::float_by_precision_t<4>>>();

  test_euqation<double, fu::complex_type_of<fu::float_by_precision_t<8>>>();

#ifdef NEWTON_FRACTAL_MPC_SUPPORT
  test_euqation<double, boostmp::mpc_complex>();
#endif

  return 0;
}