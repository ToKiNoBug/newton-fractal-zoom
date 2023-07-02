//
// Created by joseph on 6/19/23.
//

#include "object_creator.h"
#include <magic_enum.hpp>
#include "newton_equation.hpp"

#ifdef NEWTON_FRACTAL_MPC_SUPPORT
#include "mpc_support.h"
#endif

namespace newton_fractal {

tl::expected<void, std::string> is_valid_option(
    fractal_utils::float_backend_lib backend, int precision) noexcept {
  switch (backend) {
    case fu::float_backend_lib::standard: {
      switch (precision) {
        case 1:
        case 2:
          return {};
        default:
          return tl::make_unexpected(fmt::format(
              "Precision {} is invalid for cpp standard", precision));
      }
    }

    case fu::float_backend_lib::quadmath: {
#ifdef __GNUC__
      if (precision == 4) {
        return {};
      } else {
        return tl::make_unexpected("quadmath support only quad(4) precision.");
      }
#else
      return tl::make_unexpected("quadmath support is disabled.");
#endif
    }

    case fu::float_backend_lib::mpfr: {
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
      if (precision <= 0) {
        return tl::make_unexpected(
            "Precision for mpfr/mpc must be positive number.");
      }
      return {};
#else
      return tl::make_unexpected("mpc support is disabled.");
#endif
    }

    case fu::float_backend_lib::boost: {
      switch (precision) {
        case 4:
        case 8:
        case 16:
          return {};
        default:
          return tl::make_unexpected(
              fmt::format("Precision {} is invalid or unsupported for {}",
                          precision, magic_enum::enum_name(backend)));
      }
    }

    default:
      return tl::make_unexpected(fmt::format("Unsupported backend type {}",
                                             magic_enum::enum_name(backend)));
  }
}

template <typename complex_t, typename real_t>
tl::expected<complex_t, std::string> decode_complex(
    const njson& cur_p) noexcept {
  if (cur_p.size() != 2) {
    return tl::make_unexpected(
        fmt::format("Expected an array of "
                    "2 elements."));
  }
  auto temp_real = internal::decode<real_t>(cur_p[0]);
  auto temp_imag = internal::decode<real_t>(cur_p[1]);

  if (!temp_real.has_value()) {
    tl::make_unexpected(fmt::format("Failed to decode real part."));
  }
  if (!temp_imag.has_value()) {
    tl::make_unexpected(fmt::format("Failed to decode imaginary part."));
  }
  complex_t ret;
  ret.real(std::move(temp_real.value()));
  ret.imag(std::move(temp_imag.value()));

  return ret;
}

template <typename complex_t, typename real_t>
class object_creator_default_impl : public object_creator {
 public:
  using real_type = real_t;
  using complex_type = complex_t;

  [[nodiscard]] fractal_utils::float_backend_lib backend_lib()
      const noexcept override {
    return fu::backend_of<real_t>();
  }

  [[nodiscard]] tl::expected<std::unique_ptr<fractal_utils::wind_base>,
                             std::string>
  create_window(const njson& nj) const noexcept override {
    fractal_utils::center_wind<real_type> ret;
    try {
      for (size_t idx = 0; idx < 2; idx++) {
        // std::string str = .at(idx);
        auto temp = internal::decode<real_t>(nj.at("center")[idx]);
        if (!temp.has_value()) {
          return tl::make_unexpected(fmt::format(
              "Failed to decode center component at index {}", idx));
        }
        ret.center[idx] = std::move(temp.value());
      }

      {
        auto temp = internal::decode<real_t>(nj.at("x_span"));
        if (!temp.has_value()) {
          return tl::make_unexpected(
              fmt::format("Failed to decode center x_span"));
        }
        ret.x_span = std::move(temp.value());
      }
      {
        auto temp = internal::decode<real_t>(nj.at("y_span"));
        if (!temp.has_value()) {
          return tl::make_unexpected(
              fmt::format("Failed to decode center y_span"));
        }
        ret.y_span = std::move(temp.value());
      }
    } catch (std::exception& e) {
      return tl::make_unexpected(
          fmt::format("Failed to parse json. Detail: {}", e.what()));
    }

    auto* temp = new fractal_utils::center_wind<real_type>{std::move(ret)};
    std::unique_ptr<fractal_utils::wind_base> r{temp};
    return r;
  }

  [[nodiscard]] tl::expected<njson, std::string> save_window(
      const fractal_utils::wind_base& wb) const noexcept override {
    if (!wb.float_type_matches<real_type>()) {
      return tl::make_unexpected(
          fmt::format("Type of floating point number mismatch."));
    }

    const auto& wind =
        dynamic_cast<const fractal_utils::center_wind<real_type>&>(wb);

    std::vector<uint8_t> bin;
    std::string hex;
    njson ret;

    {
      njson::array_t center;
      center.resize(2);
      for (size_t i = 0; i < 2; i++) {
        internal::encode_float_to_hex(wind.center[i], bin, hex);
        center[i] = hex;
      }
      ret.emplace("center", std::move(center));
    }

    internal::encode_float_to_hex(wind.x_span, bin, hex);
    ret.emplace("x_span", hex);

    internal::encode_float_to_hex(wind.y_span, bin, hex);
    ret.emplace("y_span", hex);

    return ret;
  }

 protected:
  [[nodiscard]] tl::expected<std::vector<complex_type>, std::string>
  decode_points(const njson& nj) const noexcept {
    std::vector<complex_type> points;

    try {
      const size_t num_points = nj.size();
      if (num_points <= 1) {
        return tl::make_unexpected(fmt::format(
            "Too few points! Requires more or euqal than 2 points."));
      }
      if (num_points > 255) {
        return tl::make_unexpected(
            fmt::format("Too many points! Expected no more than 255"));
      }
      points.resize(num_points);

      for (size_t i = 0; i < num_points; i++) {
        const auto& cur_p = nj[i];
        auto cplx_opt = decode_complex<complex_type, real_type>(cur_p);
        if (!cplx_opt.has_value()) {
          return tl::make_unexpected(
              fmt::format("Failed to decode the {}-th point. Detail: {}", i,
                          cplx_opt.error()));
        }
        points[i] = std::move(cplx_opt.value());
      }
    } catch (std::exception& e) {
      return tl::make_unexpected(fmt::format(
          "Exception occurred when parsing json. Detail: {}", e.what()));
    }

    return points;
  }
};

template <int prec>
class object_creator_by_prec
    : public object_creator_default_impl<
          fu::complex_type_of<fu::float_by_precision_t<prec>>,
          fu::float_by_precision_t<prec>> {
 public:
  using complex_type = fu::complex_type_of<fu::float_by_precision_t<prec>>;

  object_creator_by_prec() = default;
  object_creator_by_prec(const object_creator_by_prec&) noexcept = default;
  object_creator_by_prec(object_creator_by_prec&&) noexcept = default;

  [[nodiscard]] std::unique_ptr<object_creator> copy() const noexcept override {
    return std::make_unique<object_creator_by_prec>(*this);
  }

  [[nodiscard]] int precision() const noexcept override { return prec; }

  [[nodiscard]] bool is_fixed_precision() const noexcept override {
    return true;
  }

  void set_precision(int _prec) & noexcept final {
    assert(this->precision() == _prec);
  }

  [[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
  create_equation(const njson& nj) const noexcept override {
    auto points = this->decode_points(nj);
    if (!points.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to decode points. Detail: {}", points.error()));
    }

    newton_equation_base* eqp = new equation_fixed_prec<prec>{points.value()};
    return std::unique_ptr<newton_equation_base>{eqp};
  }

  tl::expected<void, std::string> set_precision(
      fractal_utils::wind_base&) const noexcept override {
    return tl::make_unexpected(fmt::format(
        "Can not set precision. Precision is fixed at {}.", this->precision()));
  }

  tl::expected<void, std::string> set_precision(
      newton_equation_base&) const noexcept override {
    return tl::make_unexpected(fmt::format(
        "Can not set precision. Precision is fixed at {}.", this->precision()));
  }
};

#ifdef NEWTON_FRACTAL_MPC_SUPPORT

class object_creator_mpc
    : public object_creator_default_impl<boostmp::mpc_complex,
                                         boostmp::mpfr_float> {
 private:
  const int _precision{0};

 public:
  explicit object_creator_mpc(int precision) : _precision{precision} {}
  object_creator_mpc(const object_creator_mpc&) noexcept = default;
  object_creator_mpc(object_creator_mpc&&) noexcept = default;

  using base_t =
      object_creator_default_impl<boostmp::mpc_complex, boostmp::mpfr_float>;

  [[nodiscard]] std::unique_ptr<object_creator> copy() const noexcept override {
    return std::make_unique<object_creator_mpc>(*this);
  }

  void set_precision(int prec) & noexcept final { this->_precision = prec; }
  [[nodiscard]] int precision() const noexcept override {
    return this->_precision;
  }

  [[nodiscard]] bool is_fixed_precision() const noexcept override {
    return false;
  }

  [[nodiscard]] tl::expected<std::unique_ptr<fractal_utils::wind_base>,
                             std::string>
  create_window(const njson& nj) const noexcept override {
    auto ret = base_t::create_window(nj);
    if (ret.has_value()) {
      auto windp =
          dynamic_cast<fractal_utils::center_wind<typename base_t::real_type>*>(
              ret.value().get());
      windp->x_span.precision(this->precision());
      windp->y_span.precision(this->precision());
      for (auto& val : windp->center) {
        val.precision(this->precision());
      }
    }

    return ret;
  }

  [[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
  create_equation(const njson& nj) const noexcept override {
    auto points = this->decode_points(nj);
    if (!points.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to decode points. Detail: {}", points.error()));
    }

    newton_equation_base* eqp =
        new newton_equation_mpc{points.value(), this->precision()};
    return std::unique_ptr<newton_equation_base>{eqp};
  }

  tl::expected<void, std::string> set_precision(
      fractal_utils::wind_base& wind) const noexcept override {
    if (!wind.float_type_matches<real_type>()) {
      return tl::make_unexpected(
          fmt::format("Can not set precision. Floating type mismatch."));
    }

    auto& cwind = dynamic_cast<fractal_utils::center_wind<real_type>&>(wind);
    cwind.x_span.precision(this->precision());
    cwind.y_span.precision(this->precision());
    cwind.center[0].precision(this->precision());
    cwind.center[1].precision(this->precision());
    return {};
  }

  tl::expected<void, std::string> set_precision(
      newton_equation_base& eq) const noexcept override {
    auto eqp = dynamic_cast<newton_equation_mpc*>(&eq);
    if (eqp == nullptr) {
      return tl::make_unexpected(
          "Can not set precision. The floating type is not mpc");
    }
    eqp->set_precision(this->precision());
    return {};
  }
};

#endif

tl::expected<std::unique_ptr<object_creator>, std::string>
object_creator::create(fractal_utils::float_backend_lib backend,
                       int precision) noexcept {
  if (!is_valid_option(backend, precision)) {
    return tl::make_unexpected(
        fmt::format("Unsupported option: backend = {}, precision = {}.",
                    magic_enum::enum_name(backend), precision));
  }

  switch (backend) {
    case fu::float_backend_lib::standard:
    case fu::float_backend_lib::quadmath:
    case fu::float_backend_lib::boost: {
      object_creator* ret = nullptr;
      switch (precision) {
        case 1:
          ret = new object_creator_by_prec<1>;
          break;
        case 2:
          ret = new object_creator_by_prec<2>;
          break;
        case 4:
          ret = new object_creator_by_prec<4>;
          break;
        case 8:
          ret = new object_creator_by_prec<8>;
          break;
        case 16:
          ret = new object_creator_by_prec<16>;
          break;
        default:
          return tl::make_unexpected(
              fmt::format("Unsupported option: backend = {}, precision = {}.",
                          magic_enum::enum_name(backend), precision));
      }

      return std::unique_ptr<object_creator>{ret};
    }

#ifdef NEWTON_FRACTAL_MPC_SUPPORT
    case fu::float_backend_lib::mpfr:
      return std::unique_ptr<object_creator>{new object_creator_mpc{precision}};
#endif

    default:
      return tl::make_unexpected(
          fmt::format("Unsupported option: backend = {}, precision = {}.",
                      magic_enum::enum_name(backend), precision));
  }
}

njson::array_t object_creator::save_equation(
    const newton_equation_base& neb) const noexcept {
  return neb.to_json();
}

}  // namespace newton_fractal