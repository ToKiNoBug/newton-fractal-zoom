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
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "newton_equation_base.h"
#include <sstream>

#ifdef __GNUC__
#include <quadmath.h>
#endif

namespace fu = fractal_utils;
/*
#ifdef __GNUC__
inline std::ostream& operator<<(__float128 number, std::ostream& os) noexcept {
  os << double(number);
}
#endif

*/

namespace newton_fractal {

namespace internal {

template <typename real_t>
void encode_float_to_hex(const real_t& r, std::vector<uint8_t>& bin,
                         std::string& hex) noexcept {
  bin.resize(sizeof(real_t));
  while (true) {
    const auto bytes = fu::encode_float(r, bin);
    if (bytes != 0) {
      bin.resize(bytes);
      break;
    }
    bin.resize(bin.size() * 2);
  }

  hex.resize(bin.size() * 2 + 16);
  auto len = fu::bin_2_hex(bin, hex, true);
  assert(len.has_value());
  hex.resize(len.value());
}

template <typename real_t>
std::optional<real_t> decode(const njson& nj) noexcept {
  if (nj.is_number()) {
    return real_t(double(nj));
  }
  if (nj.is_string()) {
    const std::string str = nj;
    if (str.starts_with("0x") || str.starts_with("0X")) {
      // parse as hex
      std::vector<uint8_t> bin;
      bin.resize(str.size());
      auto len = fractal_utils::hex_2_bin(str, bin);
      if (!len.has_value()) return std::nullopt;
      bin.resize(len.value());
      return fractal_utils::decode_float<real_t>(bin);
    } else {
#ifdef __GNUC__
      constexpr bool is_quadmath = std::is_same_v<real_t, __float128>;
#else
      constexpr bool is_quadmath = false;
#endif

      if constexpr (is_quadmath) {
#ifdef __GNUC__
        char* p_end = nullptr;
        __float128 ret = strtoflt128(str.c_str(), &p_end);
        return ret;
#endif
      } else {
        thread_local std::stringstream ss;
        ss.clear();
        ss << str;
        real_t out;
        ss >> out;
        return out;
      }
    }
  }
  return std::nullopt;
}

}  // namespace internal

template <typename complex_t>
void format_complex(const complex_t& z, std::ostream& os) noexcept {
  os << z.real();
  if (z.imag() >= 0) {
    os << '+';
  }
  os << z.imag() << 'i';
}

#ifdef __GNUC__
inline void format_complex(const std::complex<__float128>& z,
                           std::ostream& os) noexcept {
  os << double(z.real());
  if (z.imag() >= 0) {
    os << '+';
  }
  os << double(z.imag()) << 'i';
}
#endif

template <typename complex_t, typename real_t>
void compute_norm2(const complex_t& a, real_t& b) noexcept {
  // complex_t temp = a * (-a);
  b = a.real() * a.real() + a.imag() * a.imag();
}

namespace internal {

inline void strip_extra_0(std::string& str) noexcept {
  while (true) {
    if (str.empty()) {
      break;
    }
    if (str.ends_with('.')) {
      break;
    }
    if (str.ends_with('0') || str.ends_with(' ')) {
      str.pop_back();
    } else {
      break;
    }
  }

  if (str.ends_with('.')) {
    str.push_back('0');
    return;
  }
  if (str.empty()) {
    str.push_back('0');
    return;
  }
}

template <typename float_t>
njson save_float_by_format(const float_t& number, float_save_format fsf,
                           std::stringstream& ss,
                           std::vector<uint8_t>& bin) noexcept {
  njson ret;
  switch (fsf) {
    case float_save_format::directly:
      ret = double(number);
      break;
    case float_save_format::hex_string: {
      std::string result;
      encode_float_to_hex(number, bin, result);
      ret = result;
      break;
    }
    case float_save_format::formatted_string: {
      ss.clear();
#ifdef __GNUC__
      constexpr bool is_quadmath = std::is_same_v<float_t, __float128>;
#else
      constexpr bool is_quadmath = false;
#endif
      ss.precision(100000);
      // ss.setf(std::ios::fixed);
      if constexpr (is_quadmath) {
        ss << double(number);
      } else {
        ss << number;
      }

      std::string result;
      ss >> result;
      strip_extra_0(result);
      ret = result;
      break;
    }
  }

  return ret;
}

}  // namespace internal

template <typename complex_t>
using point_list = std::vector<complex_t>;

template <typename complex_t, typename real_t = typename complex_t::value_type>
class newton_equation : public newton_equation_base {
 protected:
  // [a,b,c] <-> z^3 + az^2 + bz + c = 0
  std::vector<complex_t> _parameters;
  std::vector<complex_t> _points;

 public:
  newton_equation() = default;

  using complex_type = complex_t;
  using real_type = real_t;

  explicit newton_equation(std::span<const complex_t>& points) {
    this->_parameters.reserve(points.size());
    for (const auto& p : points) {
      this->add_point(p);
    }
  }

  auto& parameters() noexcept { return this->_parameters; }

  auto& parameters() const noexcept { return this->_parameters; }

  void clear() & noexcept override {
    this->_points.clear();
    this->_parameters.clear();
  }

  [[nodiscard]] inline int order() const noexcept override {
    return this->_parameters.size();
  }

  const auto& item_at_order(int _order) const noexcept {
    assert(_order < this->order());
    return this->parameters()[this->order() - _order - 1];
  }

  [[nodiscard]] std::complex<double> point_at(int idx) const noexcept override {
    return {(double)this->_points[idx].real(),
            (double)this->_points[idx].imag()};
  }

  void add_point(const complex_t& p) noexcept {
    this->_points.emplace_back(p);
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

    std::stringstream ss;

    ss << fmt::format("z^{}", this->order());

    for (int i = 0; i < this->parameters().size(); i++) {
      const int current_order = this->order() - i - 1;
      if (current_order > 0) {
        ss << " + (";
        format_complex(this->parameters()[i], ss);
        ss << ") * z^" << current_order;
        //<< this->parameters()[i]

        /*
        fmt::format_to(std::back_inserter(ret), " + ({}+{}i) * z^{}",
                       double(this->parameters()[i].real()),
                       double(this->parameters()[i].imag()), current_order);
        */
      } else {
        ss << " + (";
        format_complex(this->parameters()[i], ss);
        ss << ") = 0";
        /*
        fmt::format_to(std::back_inserter(ret), " + ({}+{}i) = 0",
                       double(this->parameters()[i].real()),
                       double(this->parameters()[i].imag()));
        */
      }
    }

    return ss.str();
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

  void iterate_n(std::any& z, int n) const noexcept {
    this->iterate_n(*std::any_cast<complex_t>(&z), n);
  }

  [[nodiscard]] static inline bool is_normal(const real_t& n) noexcept {
    if (n != n) {  // nan
      return false;
    }
    //    const real_t temp = n * 0;
    //    if (temp != temp) {  // inf*0==nan
    //      return false;
    //    }
    return true;
  }

  [[nodiscard]] static inline bool is_normal(const complex_t& z) noexcept {
    return is_normal(real_t{z.real()}) && is_normal(real_t{z.imag()});
  }

  std::optional<single_result> compute_single(
      complex_t& z, int iteration_times) const noexcept {
    assert(this->_parameters.size() == this->_points.size());
    this->iterate_n(z, iteration_times);
    if (!is_normal(z)) {
      return std::nullopt;
    }

    int min_idx = -1;
    complex_t min_diff;
    real_t min_norm2{INFINITY};
    for (int idx = 0; idx < this->order(); idx++) {
      complex_t diff = z - this->_points[idx];
      real_t diff_norm2;
      compute_norm2(diff, diff_norm2);

      if (diff_norm2 < min_norm2) {
        min_idx = idx;
        min_diff = diff;
        min_norm2 = diff_norm2;
      }
    }

    return single_result{
        min_idx,
        std::complex<double>{double(min_diff.real()), double(min_diff.imag())}};
  }

  /*
  std::optional<single_result> compute_single(
      std::any& z_any, int iteration_times) const noexcept override {
    complex_t& z = *std::any_cast<complex_t>(&z_any);
    return this->compute_single(z, iteration_times);
  }
  */

  void compute(const fractal_utils::wind_base& _wind, int iteration_times,
               compute_option& opt) const noexcept override {
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
      {
        real_t temp = r * r_unit;
        temp += r0c0.imag();
        z.imag(temp);
      }
      const real_t imag_part = z.imag();
      for (int c = 0; c < (int)cols; c++) {
        {
          real_t temp = c * c_unit;
          temp += r0c0.real();
          z.real(temp);
        }
        z.imag(imag_part);

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

 public:
  void reset(std::span<const std::complex<double>> points) & noexcept override {
    this->clear();
    for (auto& point : points) {
      complex_t temp{point};
      this->add_point(temp);
    }
  }

  [[nodiscard]] njson::array_t to_json(
      float_save_format fsf) const noexcept override {
    njson::array_t ret;
    ret.reserve(this->order());

    std::vector<uint8_t> bin;
    std::stringstream ss;
    // std::string hex;

    for (const auto& cplx : this->_points) {
      njson ::array_t temp;
      temp.reserve(2);
      // internal::encode_float_to_hex<real_type>(cplx.real(), bin, hex);
      temp.emplace_back(
          internal::save_float_by_format<real_type>(cplx.real(), fsf, ss, bin));
      // internal::encode_float_to_hex<real_type>(cplx.imag(), bin, hex);
      temp.emplace_back(
          internal::save_float_by_format<real_type>(cplx.imag(), fsf, ss, bin));

      ret.emplace_back(std::move(temp));
    }
    return ret;
  }
};

template <int prec>
class equation_fixed_prec
    : public newton_equation<
          fu::complex_type_of<fu::float_by_precision_t<prec>>,
          fu::float_by_precision_t<prec>> {
 public:
  [[nodiscard]] int precision() const noexcept { return prec; }
  using base_t =
      newton_equation<fu::complex_type_of<fu::float_by_precision_t<prec>>,
                      fu::float_by_precision_t<prec>>;

  using real_type = fu::float_by_precision_t<prec>;
  using complex_type = fu::complex_type_of<fu::float_by_precision_t<prec>>;

  equation_fixed_prec() = default;
  equation_fixed_prec(const equation_fixed_prec&) = default;
  equation_fixed_prec(equation_fixed_prec&&) noexcept = default;
  explicit equation_fixed_prec(std::span<const complex_type> points)
      : base_t{points} {}

  ~equation_fixed_prec() = default;

  [[nodiscard]] std::unique_ptr<newton_equation_base> copy()
      const noexcept override {
    return std::make_unique<equation_fixed_prec>(*this);
  }

  [[nodiscard]] static consteval bool is_fixed_precision() noexcept {
    return true;
  }
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_HPP
