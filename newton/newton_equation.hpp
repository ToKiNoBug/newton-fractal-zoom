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

template <typename complex_t>
class newton_equation : public newton_equation_base {
 private:
  // [a,b,c] <-> z^3 + az^2 + bz + c = 0
  std::vector<complex_t> _parameters;

 public:
  newton_equation() = default;

  using complex_type = complex_t;

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
};

}  // namespace newton_fractal
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
#include <boost/multiprecision/mpc.hpp>
namespace newton_fractal {

template <>
class newton_equation<boostmp::mpc_complex> : public newton_equation_base {
 private:
  std::vector<boostmp::mpc_complex> _parameters;

  struct buffer_t {
    std::array<boostmp::mpc_complex, 4> complex_arr;
    std::array<boostmp::mpfr_float, 2> real_arr;

    buffer_t() = default;
    explicit buffer_t(int prec) { this->set_precision(prec); }

    void set_precision(int prec) & noexcept {
      for (auto& val : this->real_arr) {
        val.precision(prec);
      }
      for (auto& val : this->complex_arr) {
        val.precision(prec);
      }
    }
  };

  // buffer_t buffer;

  static buffer_t& buffer() noexcept;

 public:
  newton_equation() = default;
  newton_equation(const newton_equation&) = default;
  newton_equation(newton_equation&&) = default;
  explicit newton_equation(std::span<boostmp::mpc_complex> points);
  newton_equation(std::span<boostmp::mpc_complex> points, int precision);

  ~newton_equation() override = default;

  using complex_type = boostmp::mpc_complex;

  [[nodiscard]] std::optional<int> precision() const noexcept;

  void set_precision(int p) & noexcept;

  [[nodiscard]] int order() const noexcept override {
    return (int)this->_parameters.size();
  }

  [[nodiscard]] auto& parameters() noexcept { return this->_parameters; }

  [[nodiscard]] auto& parameters() const noexcept { return this->_parameters; }

  [[nodiscard]] const auto& item_at_order(int _order) const noexcept {
    assert(_order < this->order());
    return this->parameters()[this->order() - _order - 1];
  }

  void add_point(const boostmp::mpc_complex& point) & noexcept;

  [[nodiscard]] std::string to_string() const noexcept override;

  void compute_difference(const complex_type& z,
                          complex_type& dst) const noexcept;

  [[nodiscard]] complex_type compute_difference(
      const complex_type& z) const noexcept {
    complex_type ret;
    this->compute_difference(z, ret);
    return ret;
  }

  void iterate_inplace(complex_type& z) const noexcept;
  [[nodiscard]] complex_type iterate(const complex_type& z) const noexcept;

  void iterate_n(complex_type& z, int n) const noexcept;

  void iterate_n(std::any& z, int n) const noexcept override {
    this->iterate_n(*std::any_cast<complex_type>(&z), n);
  }
};

}  // namespace newton_fractal

#endif

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP
