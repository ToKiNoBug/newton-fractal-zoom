
#include "mpc_support.h"
#include <sstream>

namespace nf = newton_fractal;
using nf::newton_equation_mpc;

newton_equation_mpc::newton_equation_mpc(int precision) {
  this->_precision = precision;
}

newton_equation_mpc::newton_equation_mpc(
    std::span<const boostmp::mpc_complex> points) {
  this->_parameters.reserve(points.size());
  for (const auto& p : points) {
    this->add_point(p);
  }
  this->update_precision();
}

newton_equation_mpc::newton_equation_mpc(
    std::span<const boostmp::mpc_complex> points, int precision)
    : newton_equation_mpc{points} {
  this->set_precision(precision);
}

int newton_equation_mpc::precision() const noexcept { return this->_precision; }

void newton_equation_mpc::set_precision(int p) & noexcept {
  assert(p > 0);
  this->_precision = p;
  // this->buffer.set_precision(p);
}

void newton_equation_mpc::update_precision() & noexcept {
  for (auto& val : this->_parameters) {
    val.precision(this->_precision);
  }
  for (auto& val : this->_points) {
    val.precision(this->_precision);
  }
}

void newton_equation_mpc::add_point(const boostmp::mpc_complex& p) & noexcept {
  const int order_before = this->order();
  this->_points.emplace_back(p);

  if (order_before > 0) {
    // temp = this->parameters().back() * (-p);
    this->parameters().emplace_back(0);
  } else {
    this->parameters().emplace_back(-p);
    return;
  }

  complex_type temp;
  for (int i = order_before - 1; i >= 0; i--) {
    temp = this->parameters()[i];
    temp *= -p;
    this->parameters()[i + 1] += temp;
  }
  this->parameters()[0] -= p;

  this->update_precision();
}

void format_complex(const boostmp::mpc_complex& z,
                    std::stringstream& ss) noexcept {
  ss << z.real();
  if (z.imag() >= 0) {
    ss << '+';
  }
  ss << z.imag() << 'i';
}

newton_equation_mpc::buffer_t& newton_equation_mpc::buffer() noexcept {
  thread_local buffer_t buf;
  return buf;
}

void newton_equation_mpc::compute_difference(const complex_type& z,
                                             complex_type& dst) const noexcept {
  // dst = 0;
  dst = this->item_at_order(0);
  auto& z_power = buffer().complex_arr[0];
  z_power = z;

  auto& temp = buffer().complex_arr[1];
  for (int o = 1; o < this->order(); o++) {
    temp = z_power * this->item_at_order(o);
    dst += temp;
    z_power *= z;
  }

  dst += z_power;
}

void newton_equation_mpc::iterate_inplace(complex_type& z) const noexcept {
  assert(this->order() > 1);
  auto& buf = this->buffer();

  auto& f = buf.complex_arr[0];
  auto& df = buf.complex_arr[1];

  auto& z_power = buf.complex_arr[2];
  auto& temp = buf.complex_arr[3];

  f = this->item_at_order(0);
  df = this->item_at_order(1);

  z_power = z;

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
  f /= df;
  f *= -1;
  z += f;
}

newton_equation_mpc::complex_type newton_equation_mpc::iterate(
    const complex_type& z) const noexcept {
  complex_type temp{z};
  this->iterate_inplace(temp);
  return temp;
}

void newton_equation_mpc::iterate_n(complex_type& z, int n) const noexcept {
  for (int i = 0; i < n; i++) {
    this->iterate_inplace(z);
  }
}

auto newton_equation_mpc::compute_single(complex_type& z,
                                         int iteration_times) const noexcept
    -> std::optional<single_result> {
  assert(this->_parameters.size() == this->_points.size());
  this->iterate_n(z, iteration_times);

  if (z.real() != z.real() || z.imag() != z.imag()) {
    return std::nullopt;
  }

  int min_idx = -1;
  complex_type min_diff;
  complex_type min_norm2{FP_INFINITE};
  for (int idx = 0; idx < this->order(); idx++) {
    complex_type diff = z - this->_points[idx];
    complex_type diff_norm2;
    compute_norm2(diff, diff_norm2);

    if (diff_norm2.real() < min_norm2.real()) {
      min_idx = idx;
      min_diff = diff;
      min_norm2 = diff_norm2;
    }
  }

  return single_result{min_idx, std::complex<double>{double(min_diff.real()),
                                                     double(min_diff.imag())}};
}

/*
auto newton_equation_mpc::compute_single(std::any& z_any,
                                         int iteration_times) const noexcept
    -> std::optional<single_result> {
  complex_type& z = *std::any_cast<complex_type>(&z_any);
  return this->compute_single(z, iteration_times);
}

*/
void newton_equation_mpc::compute(const fractal_utils::wind_base& _wind,
                                  int iteration_times,
                                  compute_option& opt) const noexcept {
  assert(opt.bool_has_result.rows() == opt.f64complex_difference.rows());
  assert(opt.f64complex_difference.rows() == opt.u8_nearest_point_idx.rows());
  const size_t rows = opt.bool_has_result.rows();

  assert(opt.bool_has_result.cols() == opt.f64complex_difference.cols());
  assert(opt.f64complex_difference.cols() == opt.u8_nearest_point_idx.cols());
  const size_t cols = opt.bool_has_result.cols();

  const auto& wind =
      dynamic_cast<const fractal_utils::center_wind<boostmp::mpfr_float>&>(
          _wind);

  const auto left_top_corner = wind.left_top_corner();
  const complex_type r0c0{left_top_corner[0], left_top_corner[1]};

  const boostmp::mpfr_float r_unit = -wind.y_span / rows;
  const boostmp::mpfr_float c_unit = wind.x_span / cols;

#pragma omp parallel for schedule(guided) default(none) \
    shared(rows, cols, r_unit, c_unit, r0c0, iteration_times, opt)
  for (int r = 0; r < (int)rows; r++) {
    thread_local complex_type z;
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
