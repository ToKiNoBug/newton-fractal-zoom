//
// Created by joseph on 6/18/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_MPC_SUPPORT_H
#define NEWTON_FRACTAL_ZOOM_MPC_SUPPORT_H

#include <core_utils.h>
#include <optional>
#include <tl/expected.hpp>
#include <string>
#include "newton_equation.hpp"

#include <boost/multiprecision/mpc.hpp>
namespace newton_fractal {

class newton_equation_mpc
    : public newton_equation<boostmp::mpc_complex, boostmp::mpfr_float> {
 protected:
  int _precision{50};

  struct buffer_t {
    std::array<boostmp::mpc_complex, 5> complex_arr;
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

  static buffer_t& buffer() noexcept;
  void update_precision() & noexcept;

 public:
  newton_equation_mpc() = default;
  explicit newton_equation_mpc(int precision);
  explicit newton_equation_mpc(std::span<const boostmp::mpc_complex> points);
  newton_equation_mpc(std::span<const boostmp::mpc_complex> points,
                      int precision);

  [[nodiscard]] static consteval bool is_fixed_precision() noexcept {
    return false;
  }

  [[nodiscard]] std::unique_ptr<newton_equation_base> copy()
      const noexcept override {
    return std::make_unique<newton_equation_mpc>(*this);
  }

  [[nodiscard]] int precision() const noexcept;

  void set_precision(int p) & noexcept;

  void clear() & noexcept override {
    this->_points.clear();
    this->_parameters.clear();
  }

  void add_point(const boostmp::mpc_complex& point) & noexcept;

  void compute_difference(const complex_type& z,
                          complex_type& dst) const noexcept;

  [[nodiscard]] complex_type compute_difference(
      const complex_type& z) const noexcept {
    complex_type ret;
    this->compute_difference(z, ret);
    return ret;
  }
  void iterate_inplace(complex_type& z, buffer_t& buf) const noexcept;

  [[nodiscard]] complex_type iterate(const complex_type& z) const noexcept;

  void iterate_n(complex_type& z, int n, buffer_t& buf) const noexcept;
  void iterate_n(complex_type& z, int n) const noexcept;

  /*
  void iterate_n(std::any& z, int n) const noexcept {
    this->iterate_n(*std::any_cast<complex_type>(&z), n);
  }
  */

  std::optional<single_result> compute_single(complex_type& z,
                                              int iteration_times,
                                              buffer_t& buf) const noexcept;

  /*
  std::optional<single_result> compute_single(
      std::any& z_any, int iteration_times) const noexcept override;
  */

  void compute(const fractal_utils::wind_base& _wind, int iteration_times,
               compute_option& opt) const noexcept override;
};

void mpc_mul_inplace_buffered(mpc_ptr z1, mpc_srcptr z2, mpc_rnd_t rnd,
                              mpc_ptr buf);
void mpc_mul_buffered(mpc_ptr dst, mpc_srcptr z1, mpc_srcptr z2, mpc_rnd_t rnd,
                      mpc_ptr buf) noexcept;

/*
void mpc_div_buffered(mpc_ptr dst, mpc_srcptr a, mpc_srcptr b, mpc_rnd_t rnd,
                      mpc_ptr buf) noexcept;
*/
void mpc_div_inplace_buffered(mpc_ptr z1, mpc_srcptr z2, mpc_rnd_t rnd,
                              mpc_ptr buf);

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_MPC_SUPPORT_H
