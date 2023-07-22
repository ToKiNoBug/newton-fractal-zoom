//
// Created by David on 2023/7/22.
//

#ifndef NEWTON_FRACTAL_ZOOM_COMPUTATION_HPP
#define NEWTON_FRACTAL_ZOOM_COMPUTATION_HPP

#include "newton_equation_base.h"

#ifdef __CUDACC__
#define NF_HOST_DEVICE_FUN __host__ __device__
#else
#define NF_HOST_DEVICE_FUN
#endif

namespace newton_fractal::internal {

template <typename float_t, typename complex_t>
struct compute_functions {
  using real_t = float_t;
  NF_HOST_DEVICE_FUN static void compute_norm2(const complex_t& a,
                                               real_t& b) noexcept {
    // complex_t temp = a * (-a);
    b = a.real() * a.real() + a.imag() * a.imag();
  }

  // template <typename complex_t>
  NF_HOST_DEVICE_FUN static const auto& item_at_order(
      const complex_t* parameters, int order, int _order) noexcept {
    assert(_order < order);
    return parameters[order - _order - 1];
  }

  static const auto& item_at_order(std::span<const complex_t> parameters,
                                   int _order) noexcept {
    return item_at_order(parameters.data(), parameters.size(), _order);
  }

  // template <typename float_t, typename complex_t>
  NF_HOST_DEVICE_FUN static complex_t iterate(const complex_t* parameters,
                                              const int order,
                                              const complex_t& z) noexcept {
    const int this_order = order;
    assert(this_order > 1);

    complex_t f{item_at_order(parameters, order, 0)},
        df{item_at_order(parameters, order, 1)};

    complex_t z_power = z;
    complex_t temp;
    for (int n = 1; n < this_order; n++) {
      {
        temp = item_at_order(parameters, order, n) * z_power;
        f += temp;
      }
      {
        if (n + 1 >= this_order) {
          temp = z_power;
          temp *= (n + 1);
        } else {
          temp = item_at_order(parameters, order, n + 1) * z_power;
          temp *= (n + 1);
        }
        df += temp;
      }
      z_power *= z;
    }

    f += z_power;

    return z - (f / df);
  }

  static complex_t iterate(std::span<const complex_t> parameters,
                           const complex_t& z) noexcept {
    return iterate(parameters.data(), parameters.size(), z);
  }

  // template <typename float_t, typename complex_t>
  NF_HOST_DEVICE_FUN static void iterate_n(const complex_t* parameters,
                                           int order, complex_t& z,
                                           int n) noexcept {
    assert(n >= 0);
    for (int i = 0; i < n; i++) {
      z = iterate(parameters, order, z);
    }
  }

  NF_HOST_DEVICE_FUN static void iterate_n(
      std::span<const complex_t> parameters, complex_t& z, int n) noexcept {
    iterate_n(parameters.data(), parameters.size(), z, n);
  }

  // template <typename real_t>
  [[nodiscard]] NF_HOST_DEVICE_FUN static inline bool is_normal_real(
      const real_t& n) noexcept {
    if (n != n) {  // nan
      return false;
    }
    //    const real_t temp = n * 0;
    //    if (temp != temp) {  // inf*0==nan
    //      return false;
    //    }
    return true;
  }

  // template <typename real_t, typename complex_t>
  [[nodiscard]] NF_HOST_DEVICE_FUN static inline bool is_normal_complex(
      const complex_t& z) noexcept {
    return is_normal_real(real_t{z.real()}) && is_normal_real(real_t{z.imag()});
  }

  // template <typename real_t, typename complex_t>
  NF_HOST_DEVICE_FUN static bool compute_single(
      const complex_t* parameters, const complex_t* points, const int order,
      complex_t& z, int iteration_times,
      newton_equation_base::single_result& result_dest) noexcept {
    iterate_n(parameters, order, z, iteration_times);
    if (!is_normal_complex(z)) {
      return false;
    }
    // const int order = parameters.size();

    int min_idx = -1;
    complex_t min_diff;
    real_t min_norm2{INFINITY};
    for (int idx = 0; idx < order; idx++) {
      complex_t diff = z - points[idx];
      real_t diff_norm2;
      compute_norm2(diff, diff_norm2);

      if (diff_norm2 < min_norm2) {
        min_idx = idx;
        min_diff = diff;
        min_norm2 = diff_norm2;
      }
    }

    result_dest.nearest_point_idx = min_idx;
    result_dest.difference =
        std::complex<double>{double(min_diff.real()), double(min_diff.imag())};

    return true;
  }

  static bool compute_single(
      std::span<const complex_t> parameters, std::span<const complex_t> points,
      complex_t& z, int iteration_times,
      newton_equation_base::single_result& result_dest) noexcept {
    assert(parameters.size() == points.size());

    return compute_single(parameters.data(), points.data(), points.size(), z,
                          iteration_times, result_dest);
  }
};

}  // namespace newton_fractal::internal

#endif  // NEWTON_FRACTAL_ZOOM_COMPUTATION_HPP
