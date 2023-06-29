
#include "mpc_support.h"
#include <sstream>

namespace nf = newton_fractal;
using nf::newton_equation_mpc;

void nf::mpc_mul_inplace_buffered(mpc_ptr z1, mpc_srcptr z2, mpc_rnd_t rnd,
                                  mpc_ptr buf) {
  assert(z1 != z2);
  assert(z1 != buf);
  assert(z2 != buf);

  // mpf_class &a = this->_real;
  mpfr_ptr a = mpc_realref(z1);
  // mpf_class &b = this->_imag;
  mpfr_ptr b = mpc_imagref(z1);
  // const mpf_class &c = Z._real;
  mpfr_srcptr c = mpc_realref(z2);
  // const mpf_class &d = Z._imag;
  mpfr_srcptr d = mpc_imagref(z2);

  // mpf_class &bd = buf.float_arr[0];
  mpfr_ptr bd = mpc_realref(buf);
  // mpf_class &ad = buf.float_arr[1];
  mpfr_ptr ad = mpc_imagref(buf);

  // bd = b * d;
  mpfr_mul(bd, b, d, MPC_RND_IM(rnd));

  // ad = a * d;
  mpfr_mul(ad, a, d, MPC_RND_RE(rnd));

  // a->ac, b-> bc
  // a *= Z._real;
  mpfr_mul(a, a, c, MPC_RND_RE(rnd));
  // b *= Z._real;
  mpfr_mul(b, b, c, MPC_RND_IM(rnd));

  // a -= bd;
  mpfr_sub(a, a, bd, MPC_RND_RE(rnd));
  // b += ad;
  mpfr_add(b, b, ad, MPC_RND_IM(rnd));
}

void nf::mpc_mul_buffered(mpc_ptr dst, mpc_srcptr z1, mpc_srcptr z2,
                          mpc_rnd_t rnd, mpc_ptr buf) noexcept {
  assert(z1 != dst);
  assert(z2 != dst);
  assert(buf != dst);
  assert(buf != z1);
  assert(buf != z2);

  mpfr_srcptr a = mpc_realref(z1);
  mpfr_srcptr b = mpc_imagref(z1);
  mpfr_srcptr c = mpc_realref(z2);
  mpfr_srcptr d = mpc_imagref(z2);

  mpfr_mul(mpc_realref(dst), a, c, MPC_RND_RE(rnd));
  mpfr_mul(mpc_imagref(dst), b, c, MPC_RND_IM(rnd));

  mpfr_ptr bd = mpc_realref(buf);
  mpfr_ptr ad = mpc_imagref(buf);
  mpfr_mul(bd, b, d, MPC_RND_RE(rnd));
  mpfr_mul(ad, a, d, MPC_RND_IM(rnd));

  mpfr_sub(mpc_realref(dst), mpc_realref(dst), bd, MPC_RND_RE(rnd));
  mpfr_add(mpc_imagref(dst), mpc_imagref(dst), ad, MPC_RND_IM(rnd));
}

/*
void nf::mpc_div_buffered(mpc_ptr dst, mpc_srcptr a, mpc_srcptr b,
                          mpc_rnd_t rnd, mpc_ptr buf) noexcept {}
*/

void nf::mpc_div_inplace_buffered(mpc_ptr z1, mpc_srcptr z2, mpc_rnd_t rnd,
                                  mpc_ptr buf) {
  assert(z1 != z2);
  assert(buf != z1);
  assert(buf != z2);
  // mpf_class &a = this->_real;
  // mpf_class &b = this->_imag;
  mpfr_ptr a = mpc_realref(z1);
  mpfr_ptr b = mpc_imagref(z1);

  // const mpf_class& c = Z._real;
  // const mpf_class& d = Z._imag;
  mpfr_srcptr c = mpc_realref(z2);
  mpfr_srcptr d = mpc_imagref(z2);

  {
    // mpf_class& bd = buf.float_arr[0];
    // mpf_class& ad = buf.float_arr[1];
    mpfr_ptr bd = mpc_realref(buf);
    mpfr_ptr ad = mpc_imagref(buf);

    // bd = b * d;
    mpfr_mul(bd, b, d, MPC_RND_IM(rnd));

    // ad = a * d;
    mpfr_mul(ad, a, d, MPC_RND_RE(rnd));

    // a->ac, b-> bc
    // a *= c;
    // b *= c;
    mpfr_mul(a, a, c, MPC_RND_RE(rnd));
    mpfr_mul(b, b, c, MPC_RND_IM(rnd));

    // a += bd;
    // b -= ad;
    mpfr_add(a, a, ad, MPC_RND_RE(rnd));
    mpfr_sub(b, b, ad, MPC_RND_IM(rnd));
  }
  {
    // auto& c2 = buf.float_arr[0];
    // auto& d2 = buf.float_arr[1];
    mpfr_ptr c2 = mpc_realref(buf);
    mpfr_ptr d2 = mpc_imagref(buf);

    // c2 = c * c;
    mpfr_mul(c2, c, c, MPC_RND_RE(rnd));
    // d2 = d * d;
    mpfr_mul(d2, d, d, MPC_RND_IM(rnd));
    // c2 += d2;  // c2-> c2+d2
    mpfr_add(c2, c2, d2, MPC_RND_RE(rnd));

    // a /= c2;
    // b /= c2;
    mpfr_div(a, a, c2, MPC_RND_RE(rnd));
    mpfr_div(b, b, c2, MPC_RND_IM(rnd));
  }
}

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
  auto& extra_buf = buf.complex_arr[4];
  assert(f.precision() == df.precision());
  assert(df.precision() == z_power.precision());
  assert(temp.precision() == z.precision());
  assert(z.precision() == extra_buf.precision());

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
      /*mpc_mul(temp.backend().data(), z_power.backend().data(),
              this->item_at_order(n).backend().data(), MPC_RNDNN);*/

      mpc_mul_buffered(temp.backend().data(), z_power.backend().data(),
                       this->item_at_order(n).backend().data(), MPC_RNDNN,
                       extra_buf.backend().data());

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
        mpc_mul_buffered(temp.backend().data(), z_power.backend().data(),
                         this->item_at_order(n + 1).backend().data(), MPC_RNDNN,
                         extra_buf.backend().data());
        // temp *= (n + 1);
        mpc_mul_ui(temp.backend().data(), temp.backend().data(), n + 1,
                   MPC_RNDNN);
      }
      // df += temp;
      mpc_add(df.backend().data(), df.backend().data(), temp.backend().data(),
              MPC_RNDNN);
    }
    // z_power *= z;
    mpc_mul_inplace_buffered(z_power.backend().data(), z.backend().data(),
                             MPC_RNDNN, extra_buf.backend().data());
  }

  // f += z_power;
  mpc_add(f.backend().data(), f.backend().data(), z_power.backend().data(),
          MPC_RNDNN);

  // f /= df;
  /*mpc_div(f.backend().data(), f.backend().data(), df.backend().data(),
          MPC_RNDNN);*/

  mpc_div_inplace_buffered(f.backend().data(), df.backend().data(), MPC_RNDNN,
                           extra_buf.backend().data());

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

bool is_limited(mpfr_srcptr a) noexcept {
  return !(mpfr_nan_p(a) || mpfr_inf_p(a));
}

bool is_limited(mpc_srcptr z) noexcept {
  return is_limited(mpc_realref(z)) && is_limited(mpc_imagref(z));
}

auto newton_equation_mpc::compute_single(complex_type& z, int iteration_times,
                                         buffer_t& buf) const noexcept
    -> std::optional<single_result> {
  assert(this->_parameters.size() == this->_points.size());
  this->iterate_n(z, iteration_times, buf);

  if (!is_limited(z.backend().data())) {
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
      mpc_swap(diff.backend().data(), min_diff.backend().data());
      // mpc_set(min_diff.backend().data(), diff.backend().data(), MPC_RNDNN);
      //  min_norm2 = diff_norm2;
      // mpfr_set(min_norm2.backend().data(),
      // diff_norm2.backend().data(),MPFR_RNDN);
      mpfr_swap(min_norm2.backend().data(), diff_norm2.backend().data());
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

  auto compute_part_function = [this](mpfr_srcptr unit, int idx,
                                      mpfr_srcptr offset, mpfr_ptr dest) {
    mpfr_mul_ui(dest, unit, idx, MPFR_RNDN);
    mpfr_add(dest, dest, unit, MPFR_RNDN);
  };

#pragma omp parallel for schedule(guided) default(shared)
  for (int r = 0; r < (int)rows; r++) {
    thread_local complex_type z{0, 0, (uint32_t)this->_precision};
    thread_local buffer_t buf{this->_precision};

    thread_local real_type imag_part{0, (uint32_t)this->_precision};

    compute_part_function(r_unit.backend().data(), r, r0.backend().data(),
                          imag_part.backend().data());

    for (int c = 0; c < (int)cols; c++) {
      mpfr_set(mpc_imagref(z.backend().data()), imag_part.backend().data(),
               MPFR_RNDN);
      compute_part_function(c_unit.backend().data(), c, c0.backend().data(),
                            mpc_realref(z.backend().data()));

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
