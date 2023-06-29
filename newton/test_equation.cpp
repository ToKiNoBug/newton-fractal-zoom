//
// Created by David on 2023/6/17.
//

#include "newton_fractal.h"

#include "newton_equation.hpp"
#include <iostream>
#include <complex>

#include "newton_equation.hpp"
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
#include "mpc_support.h"
#endif

namespace nf = newton_fractal;
using std::cout, std::endl;

template <class eq_t, typename float_disp_t>
void test_euqation() noexcept {
  using real_t = typename eq_t::real_type;
  using complex_t = typename eq_t::complex_type;
  nf::point_list<complex_t> points;
  points.emplace_back(1, 2);
  points.emplace_back(3, 4);
  points.emplace_back(5, 6);
  //  points.emplace_back(7, 8);
  //  points.emplace_back(9, 10);
  //  points.emplace_back(11, 12);
  //  points.emplace_back(1, 2);

  eq_t ne{points};

  fmt::print("equation : {}\n", ne.to_string());

  for (auto& p : points) {
    auto diff = ne.compute_difference(p);

    complex_t norm2;
    newton_fractal::compute_norm2(diff, norm2);

    fmt::print("Difference at ({}+{}i) = {}\n", float_disp_t(p.real()),
               float_disp_t(p.imag()), float_disp_t(norm2.real()));
  }

  complex_t z = 10000;
  if constexpr (!ne.is_fixed_precision()) {
    ne.set_precision(100);
    z.precision(ne.precision());
  }
  for (int i = 0; i < 30; i++) {
    complex_t z_1 = ne.iterate(z);

    fmt::print("({}+{}i) iterates to ({}+{}i)\n", float_disp_t(z.real()),
               float_disp_t(z.imag()), float_disp_t(z_1.real()),
               float_disp_t(z_1.imag()));
    z = z_1;
  }
}

int main(int argc, char** argv) {
  test_euqation<nf::equation_fixed_prec<2>, double>();
  if (false) {
    test_euqation<nf::equation_fixed_prec<1>, float>();
    test_euqation<nf::equation_fixed_prec<4>, double>();
    test_euqation<nf::equation_fixed_prec<8>, double>();
    test_euqation<nf::equation_fixed_prec<16>, double>();
  }

#ifdef NEWTON_FRACTAL_MPC_SUPPORT
  test_euqation<nf::newton_equation_mpc, double>();
#endif

  return 0;
}