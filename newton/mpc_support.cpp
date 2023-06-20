
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

void newton_equation_mpc::iterate_inplace(complex_type& z,
                                          buffer_t& buf) const noexcept {
  assert(this->order() > 1);
  // auto& buf = this->buffer();

  auto& f = buf.complex_arr[0];
  auto& df = buf.complex_arr[1];

  auto& z_power = buf.complex_arr[2];
  auto& temp = buf.complex_arr[3];
  assert(f.precision() == df.precision());
  assert(df.precision() == z_power.precision());
  assert(temp.precision() == z.precision());

  // f = this->item_at_order(0);
  // f.precision(this->_precision);
  mpc_set(f.backend().data(), this->item_at_order(0).backend().data(),
          MPC_RNDNN);
  // df = this->item_at_order(1);
  // df.precision(this->_precision);
  mpc_set(df.backend().data(), this->item_at_order(1).backend().data(),
          MPC_RNDNN);

  // z_power = z;
  // z.precision(this->_precision);
  mpc_set(z_power.backend().data(), z.backend().data(), MPC_RNDNN);

  // temp.precision(this->_precision);
  for (int n = 1; n < this->order(); n++) {
    {
      // temp = this->item_at_order(n) * z_power;
      mpc_mul(temp.backend().data(), z_power.backend().data(),
              this->item_at_order(n).backend().data(), MPC_RNDNN);

      // f += temp;
      mpc_add(f.backend().data(), f.backend().data(), temp.backend().data(),
              MPC_RNDNN);
    }
    {
      if (n + 1 >= this->order()) {
        // temp = z_power;
        mpc_set(temp.backend().data(), z_power.backend().data(), MPC_RNDNN);
        // temp *= (n + 1);
        mpc_mul_ui(temp.backend().data(), temp.backend().data(), n + 1,
                   MPC_RNDNN);
      } else {
        // temp = this->item_at_order(n + 1) * z_power;
        mpc_mul(temp.backend().data(), z_power.backend().data(),
                this->item_at_order(n + 1).backend().data(), MPC_RNDNN);
        // temp *= (n + 1);
        mpc_mul_ui(temp.backend().data(), temp.backend().data(), n + 1,
                   MPC_RNDNN);
      }
      // df += temp;
      mpc_add(df.backend().data(), df.backend().data(), temp.backend().data(),
              MPC_RNDNN);
    }
    // z_power *= z;
    mpc_mul(z_power.backend().data(), z_power.backend().data(),
            z.backend().data(), MPC_RNDNN);
  }

  // f += z_power;
  mpc_add(f.backend().data(), f.backend().data(), z_power.backend().data(),
          MPC_RNDNN);

  // f /= df;
  mpc_div(f.backend().data(), f.backend().data(), df.backend().data(),
          MPC_RNDNN);

  // z -= f;
  mpc_sub(z.backend().data(), z.backend().data(), f.backend().data(),
          MPC_RNDNN);
}

newton_equation_mpc::complex_type newton_equation_mpc::iterate(
    const complex_type& z) const noexcept {
  complex_type temp{z};
  this->iterate_inplace(temp, buffer());
  return temp;
}

void newton_equation_mpc::iterate_n(complex_type& z, int n) const noexcept {
  this->iterate_n(z, n, buffer());
}

void newton_equation_mpc::iterate_n(complex_type& z, int n,
                                    buffer_t& buf) const noexcept {
  for (int i = 0; i < n; i++) {
    this->iterate_inplace(z, buf);
  }
}

auto newton_equation_mpc::compute_single(complex_type& z, int iteration_times,
                                         buffer_t& buf) const noexcept
    -> std::optional<single_result> {
  assert(this->_parameters.size() == this->_points.size());
  this->iterate_n(z, iteration_times, buf);

  if (z.real() != z.real() || z.imag() != z.imag()) {
    return std::nullopt;
  }

  int min_idx = -1;

  // complex_type min_diff;
  auto& min_diff = buf.complex_arr[0];

  // complex_type min_norm2{FP_INFINITE};
  auto& min_norm2 = buf.real_arr[0];
  mpfr_set_d(min_norm2.backend().data(), NAN, MPFR_RNDN);

  auto& diff = buf.complex_arr[1];
  auto& diff_norm2 = buf.real_arr[1];

  for (int idx = 0; idx < this->order(); idx++) {
    // complex_type diff = z - this->_points[idx];
    mpc_sub(diff.backend().data(), z.backend().data(),
            this->_points[idx].backend().data(), MPC_RNDNN);

    // complex_type diff_norm2;
    // compute_norm2(diff, diff_norm2);
    mpc_norm(diff_norm2.backend().data(), diff.backend().data(), MPFR_RNDN);

    if (diff_norm2.real() < min_norm2.real()) {
      min_idx = idx;

      // this may be optimized by swapping
      // min_diff = diff;
      mpc_set(min_diff.backend().data(), diff.backend().data(), MPC_RNDNN);
      // min_norm2 = diff_norm2;
      mpfr_set(min_norm2.backend().data(), diff_norm2.backend().data(),
               MPFR_RNDN);
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
  const real_type r0{left_top_corner[0], (uint32_t)this->_precision};
  const real_type c0{left_top_corner[1], (uint32_t)this->_precision};
  /*const complex_type r0c0{left_top_corner[0], left_top_corner[1],
                          (uint32_t)this->_precision};
  */

  const boostmp::mpfr_float r_unit = -wind.y_span / rows;
  const boostmp::mpfr_float c_unit = wind.x_span / cols;

  /*
#pragma omp parallel for schedule(guided) default(none) \
    shared(rows, cols, r_unit, c_unit, r0, c0, iteration_times, opt)
  */
  for (int r = 0; r < (int)rows; r++) {
    thread_local complex_type z{0, 0, (uint32_t)this->_precision};
    thread_local buffer_t buf{this->_precision};
    {
      // z.imag(r0c0.real()+r*r_unit)
      auto& imag_temp = buf.real_arr[0];
      mpfr_mul_ui(imag_temp.backend().data(), r_unit.backend().data(), r,
                  MPFR_RNDN);
      mpfr_add(mpc_imagref(z.backend().data()), imag_temp.backend().data(),
               r0.backend().data(), MPFR_RNDN);

      // mpc_imag(z.backend().data(), imag_temp.backend().data(), MPFR_RNDN);
    }
    for (int c = 0; c < (int)cols; c++) {
      {
        // z.real(r0c0.real() + c * c_unit);
        auto& real_temp = buf.real_arr[0];
        mpfr_mul_ui(real_temp.backend().data(), c_unit.backend().data(), c,
                    MPFR_RNDN);
        mpfr_add(mpc_realref(z.backend().data()), real_temp.backend().data(),
                 c0.backend().data(), MPFR_RNDN);
      }

      auto result = this->compute_single(z, iteration_times, buf);
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
