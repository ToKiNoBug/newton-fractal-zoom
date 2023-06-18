//
// Created by joseph on 6/18/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP
#define NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP

#include <core_utils.h>
#include <multiprecision_utils.h>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <complex>
#include <tl/expected.hpp>
// #include <expected>
#include <span>
#include <variant>
#include <vector>
#include <fmt/format.h>
#include <iterator>
#include <string>

#include "newton_equation_base.h"

namespace fu = fractal_utils;

namespace newton_fractal {

template <typename complex_t>
void compute_norm2(const complex_t& a, complex_t& b) noexcept {
  b = a * (-a);
}
template <typename complex_t>
using point_list = std::vector<complex_t>;

template <typename complex_t, typename real_t = complex_t::value_type>
class newton_equation : public newton_equation_base {
 private:
  // [a,b,c] <-> z^3 + az^2 + bz + c = 0
  std::vector<complex_t> _parameters;

  std::vector<complex_t> _points;

 public:
  newton_equation() = default;

  using complex_type = complex_t;
  using real_type = real_t;

  explicit newton_equation(const point_list<complex_t>& points) {
    this->_parameters.reserve(points.size());
    for (const auto& p : points) {
      this->add_point(p);
    }
  }

  auto& parameters() noexcept { return this->_parameters; }

  auto& parameters() const noexcept { return this->_parameters; }

  [[nodiscard]] inline int order() const noexcept override {
    return this->_parameters.size();
  }

  const auto& item_at_order(int _order) const noexcept {
    assert(_order < this->order());
    return this->parameters()[this->order() - _order - 1];
  }

  void add_point(const complex_t& p) noexcept {
    const int order_before = this->order();

    if (order_before > 0) {
      // temp = this->parameters().back() * (-p);
      this->parameters().emplace_back(0);
    } else {
      this->parameters().emplace_back(-p);
      return;
    }

    complex_t temp;
    for (int i = order_before - 1; i >= 0; i--) {
      temp = this->parameters()[i];
      temp *= -p;
      this->parameters()[i + 1] += temp;
    }
    this->parameters()[0] -= p;

    this->_points.emplace_back(p);
  }

  [[nodiscard]] std::string to_string() const noexcept override {
    if (this->order() == 0) {
      return "0 = 0";
    }

    std::string ret = fmt::format("z^{}", this->order());

    for (int i = 0; i < this->parameters().size(); i++) {
      const int current_order = this->order() - i - 1;
      if (current_order > 0) {
        fmt::format_to(std::back_inserter(ret), " + ({}+{}i) * z^{}",
                       double(this->parameters()[i].real()),
                       double(this->parameters()[i].imag()), current_order);
      } else {
        fmt::format_to(std::back_inserter(ret), " + ({}+{}i) = 0",
                       double(this->parameters()[i].real()),
                       double(this->parameters()[i].imag()));
      }
    }

    return ret;
  }

  void compute_difference(const complex_t& z, complex_t& dst) const noexcept {
    // dst = 0;
    dst = this->item_at_order(0);
    complex_t z_power = z;
    for (int o = 1; o < this->order(); o++) {
      dst += z_power * this->item_at_order(o);
      z_power *= z;
    }

    dst += z_power;
  }

  [[nodiscard]] complex_t compute_difference(
      const complex_t& z) const noexcept {
    complex_t ret;
    this->compute_difference(z, ret);
    return ret;
  }

  complex_t iterate(const complex_t& z) const noexcept {
    assert(this->order() > 1);

    complex_t f{this->item_at_order(0)}, df{this->item_at_order(1)};

    complex_t z_power = z;
    complex_t temp;
    for (int n = 1; n < this->order(); n++) {
      {
        temp = this->item_at_order(n) * z_power;
        f += temp;
      }
      {
        if (n + 1 >= this->order()) {
          temp = z_power;
          temp *= (n + 1);
        } else {
          temp = this->item_at_order(n + 1) * z_power;
          temp *= (n + 1);
        }
        df += temp;
      }
      z_power *= z;
    }

    f += z_power;

    return z - (f / df);
  }

  void iterate_n(complex_t& z, int n) const noexcept {
    assert(n >= 0);
    for (int i = 0; i < n; i++) {
      z = this->iterate(z);
    }
  }

  void iterate_n(std::any& z, int n) const noexcept override {
    this->iterate_n(*std::any_cast<complex_t>(&z), n);
  }

  std::optional<single_result> compute_single(
      complex_t& z, int iteration_times) const noexcept {
    assert(this->_parameters.size() == this->_points.size());
    this->iterate_n(z, iteration_times);
    if (z.real() != z.real() || z.imag() != z.imag()) {
      return std::nullopt;
    }

    int min_idx = -1;
    complex_t min_diff;
    complex_t min_norm2{FP_INFINITE};
    for (int idx = 0; idx < this->order(); idx++) {
      complex_t diff = z - this->_points[idx];
      complex_t diff_norm2;
      compute_norm2(diff, diff_norm2);

      if (diff_norm2.real() < min_norm2.real()) {
        min_idx = idx;
        min_diff = diff;
        min_norm2 = diff_norm2;
      }
    }

    return single_result{
        min_idx,
        std::complex<double>{double(min_diff.real()), double(min_diff.imag())}};
  }

  std::optional<single_result> compute_single(
      std::any& z_any, int iteration_times) const noexcept override {
    complex_t& z = *std::any_cast<complex_t>(&z_any);
    return this->compute_single(z, iteration_times);
  }

  void compute(const fractal_utils::wind_base& _wind, int iteration_times,
               compute_row_option& opt) const noexcept override {
    assert(opt.bool_has_result.rows() == opt.f64complex_difference.rows());
    assert(opt.f64complex_difference.rows() == opt.u8_nearest_point_idx.rows());
    const size_t rows = opt.bool_has_result.rows();

    assert(opt.bool_has_result.cols() == opt.f64complex_difference.cols());
    assert(opt.f64complex_difference.cols() == opt.u8_nearest_point_idx.cols());
    const size_t cols = opt.bool_has_result.cols();

    const auto& wind =
        dynamic_cast<const fractal_utils::center_wind<real_t>&>(_wind);

    const auto left_top_corner = wind.left_top_corner();
    const complex_t r0c0{left_top_corner[0], left_top_corner[1]};

    const real_t r_unit = -wind.y_span / rows;
    const real_t c_unit = wind.x_span / cols;

#pragma omp parallel for schedule(guided) default(none) \
    shared(rows, cols, r_unit, c_unit, r0c0, iteration_times, opt)
    for (int r = 0; r < (int)rows; r++) {
      complex_t z;
      z.imag(r0c0.imag() + r * r_unit);
      for (int c = 0; c < (int)cols; c++) {
        z.real(r0c0.real() + c * c_unit);

        auto result = this->compute_single(z, iteration_times);
        opt.bool_has_result.at<bool>(r, c) = result.has_value();
        if (result.has_value()) {
          opt.u8_nearest_point_idx.at<uint8_t>(r, c) =
              result.value().nearest_point_idx;
          opt.f64complex_difference.at<std::complex<double>>(r, c) =
              result.value().difference;
        } else {
          opt.u8_nearest_point_idx.at<uint8_t>(r, c) = 255;
          opt.f64complex_difference.at<std::complex<double>>(r, c) = {NAN, NAN};
        }
      }
    }
  }
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP
