//
// Created by joseph on 6/18/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP
#define NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP

#include "multiprecision_utils.h"
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

namespace fu = fractal_utils;

namespace newton_fractal {

template <typename complex_t>
void compute_norm2(const complex_t& a, complex_t& b) noexcept {
  b = a * (-a);
}

template <typename complex_t>
using point_list = std::vector<complex_t>;

template <typename complex_t>
class newton_equation {
 public:
  // [a,b,c] <-> z^3 + az^2 + bz + c = 0
  std::vector<complex_t> _parameters;

  newton_equation() = default;

  explicit newton_equation(const point_list<complex_t>& points) {
    this->_parameters.reserve(points.size());
    for (const auto& p : points) {
      this->add_point(p);
    }
  }

  auto& parameters() noexcept { return this->_parameters; }

  auto& parameters() const noexcept { return this->_parameters; }

  [[nodiscard]] inline int order() const noexcept {
    return this->_parameters.size();
  }

  const auto& item_at_order(int _order) const noexcept {
    return this->parameters()[this->order() - _order - 1];
  }

  void add_point(const complex_t& p) noexcept {
    // assert(buffer.size() >= this->parameters().size());
    const int order_before = this->order();
    const int order_new = order_before + 1;

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
  }

  template <typename enable = void>
  [[nodiscard]] std::string to_string() const noexcept {
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
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP
