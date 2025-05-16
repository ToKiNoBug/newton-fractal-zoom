//
// Created by joseph on 6/19/23.
//

#include "object_creator.h"
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include "newton_equation.hpp"
#include "cuda_support.cuh"
#include "OpenCL_support.h"

#ifdef NEWTON_FRACTAL_MPC_SUPPORT
#include "mpc_support.h"
#endif

namespace newton_fractal {

tl::expected<void, std::string> is_valid_option(
    fractal_utils::float_backend_lib backend, int precision,
    gpu_backend gpu) noexcept {
  if (gpu != gpu_backend::no) {
    if (backend != fu::float_backend_lib::standard) {
      return tl::make_unexpected(
          "GPU can only compute standard floating point types.");
    }
  }

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
#ifdef NF_USE_QUADMATH
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

  static constexpr char dynamic_precision_hex_seperator = ',';

  [[nodiscard]] fractal_utils::float_backend_lib backend_lib()
      const noexcept override {
    return fu::backend_of<real_t>();
  }

  [[nodiscard]] gpu_backend gpu() const noexcept override {
    return gpu_backend::no;
  }

  [[nodiscard]] std::optional<opencl_option_t> opencl_option()
      const noexcept override {
    return std::nullopt;
  }

  [[nodiscard]] tl::expected<void, std::string> set_opencl_option(
      const opencl_option_t& opt) & noexcept override {
    return tl::make_unexpected(
        "Calling set_opencl_option on a non-opencl object_creator.");
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
      const fractal_utils::wind_base& wb,
      float_save_format fsf) const noexcept override {
    if (!wb.float_type_matches<real_type>()) {
      return tl::make_unexpected(
          fmt::format("Type of floating point number mismatch."));
    }

    const auto& wind =
        dynamic_cast<const fractal_utils::center_wind<real_type>&>(wb);

    std::vector<uint8_t> bin;
    std::stringstream ss;
    njson ret;

    {
      njson::array_t center;
      center.resize(2);
      for (size_t i = 0; i < 2; i++) {
        // internal::encode_float_to_hex(wind.center[i], bin, hex);
        center[i] =
            internal::save_float_by_format(wind.center[i], fsf, ss, bin);
      }
      ret.emplace("center", std::move(center));
    }

    // internal::encode_float_to_hex(wind.x_span, bin, hex);
    ret.emplace("x_span",
                internal::save_float_by_format(wind.x_span, fsf, ss, bin));

    // internal::encode_float_to_hex(wind.y_span, bin, hex);
    ret.emplace("y_span",
                internal::save_float_by_format(wind.y_span, fsf, ss, bin));

    return ret;
  }

  [[nodiscard]] tl::expected<std::string, std::string> encode_centerhex(
      const fractal_utils::wind_base& wb) const noexcept override {
    if (!wb.float_type_matches<real_t>()) {
      return tl::make_unexpected(
          fmt::format("Type of floating point number mismatch."));
    }

    const auto& wind = dynamic_cast<const fu::center_wind<real_t>&>(wb);
    std::string ret{};
    ret.reserve(4096);
    std::vector<uint8_t> bin;
    bin.resize(2048);
    std::string temp;
    {
      const auto bytes = fu::encode_float(wind.center[0], bin);
      bin.resize(bytes);
      temp.resize(bytes * 2 + 16);
      auto chars = fu::bin_2_hex(bin, temp, true);
      temp.resize(chars.value());
      ret += temp;
    }
    const bool is_fixed = this->is_fixed_precision();
    {
      const auto bytes = fu::encode_float(wind.center[1], bin);
      bin.resize(bytes);
      temp.resize(bytes * 2 + 16);
      // for fixed precision, encode to a string; otherwise encode as two string
      // seperated by ','
      auto chars = fu::bin_2_hex(bin, temp, !is_fixed);
      temp.resize(chars.value());

      if (is_fixed) {
        ret += temp;
      } else {
        ret.push_back(',');
        ret += temp;
      }
    }

    return ret;
  }

  [[nodiscard]] tl::expected<void, std::string> decode_centerhex(
      std::string_view hex,
      fractal_utils::wind_base& wb) const noexcept override {
    if (!wb.float_type_matches<real_t>()) {
      return tl::make_unexpected(
          fmt::format("Type of floating point number mismatch."));
    }

    auto& wind = dynamic_cast<fu::center_wind<real_t>&>(wb);
    std::vector<uint8_t> bin;
    bin.resize(hex.size());
    {
      auto bytes = fu::hex_2_bin(hex, bin);
      bin.resize(bytes.value());
    }
    if (bin.size() % 2 != 0) {
      return tl::make_unexpected(
          fmt::format("Invaid centerhex, the decoded binary contains {} bytes, "
                      "but expected even number.",
                      bin.size()));
    }

    for (size_t idx = 0; idx < 2; idx++) {
      const size_t offset = idx * (bin.size() / 2);

      if (!this->decode_float_single({bin.data() + offset, bin.size() / 2},
                                     wind.center[idx])) {
        return tl::make_unexpected(fmt::format(
            "Failed to decode the {}-th component of center hex.", idx));
      }
    }

    return {};
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

  [[nodiscard]] bool decode_float_single(std::span<const uint8_t> bin,
                                         real_t& dst) const noexcept {
    auto res = fu::decode_float<real_t>(bin);

    if (!res.has_value()) {
      return false;
    }
    dst = std::move(res.value());
    return true;
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

  [[nodiscard]] virtual std::unique_ptr<object_creator> copy()
      const noexcept override {
    return std::make_unique<object_creator_by_prec>(*this);
  }

  [[nodiscard]] int precision() const noexcept override { return prec; }

  [[nodiscard]] bool is_fixed_precision() const noexcept override {
    return true;
  }

  void set_precision(int _prec) & noexcept final {
    assert(this->precision() == _prec);
  }

  [[nodiscard]] virtual tl::expected<std::unique_ptr<newton_equation_base>,
                                     std::string>
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

  [[nodiscard]] int suggested_precision_of(const fractal_utils::wind_base&, int,
                                           int) const noexcept final {
    return prec;
  }

  [[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
  clone_with_precision(const newton_equation_base& src,
                       int p) const noexcept final {
    if (p != this->precision()) {
      return tl::make_unexpected(
          "Invalid precision, you are calling clone_with_precision on a fixed "
          "precision instance.");
    }
    return src.copy();
  }
};

template <int precision>
class cuda_object_creator : public object_creator_by_prec<precision> {
 public:
  [[nodiscard]] gpu_backend gpu() const noexcept final {
    return gpu_backend::cuda;
  }

  [[nodiscard]] std::unique_ptr<object_creator> copy() const noexcept override {
    return std::make_unique<cuda_object_creator>(*this);
  }

  [[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
  create_equation(const njson& nj) const noexcept override {
    auto points = this->decode_points(nj);
    if (!points.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to decode points. Detail: {}", points.error()));
    }

    auto eq_exp =
        internal::create_complete_equation<fu::float_by_precision_t<precision>>(
            points.value());
    return eq_exp;
  }
};

template <int precision>
class opencl_object_creator : public object_creator_by_prec<precision> {
 private:
  opencl_option_t m_opencl_option;

 public:
  opencl_object_creator() = default;
  ~opencl_object_creator() override = default;

  [[nodiscard]] gpu_backend gpu() const noexcept final {
    return gpu_backend::opencl;
  }

  [[nodiscard]] std::unique_ptr<object_creator> copy() const noexcept override {
    return std::make_unique<opencl_object_creator>(*this);
  }

  [[nodiscard]] std::optional<opencl_option_t> opencl_option()
      const noexcept final {
    return this->m_opencl_option;
  }

  [[nodiscard]] tl::expected<void, std::string> set_opencl_option(
      const opencl_option_t& opt) & noexcept final {
    this->m_opencl_option = opt;
    return {};
  }

  [[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
  create_equation(const njson& nj) const noexcept override {
    auto points = this->decode_points(nj);
    if (!points.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to decode points. Detail: {}", points.error()));
    }

    auto eq_exp = create_opencl_equation<fu::float_by_precision_t<precision>>(
        this->m_opencl_option, points.value());
    return eq_exp;
  }
};

#ifdef NEWTON_FRACTAL_MPC_SUPPORT

class object_creator_mpc
    : public object_creator_default_impl<boostmp::mpc_complex,
                                         boostmp::mpfr_float> {
 private:
  int _precision{0};

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

  [[nodiscard]] tl::expected<void, std::string> decode_centerhex(
      std::string_view hex, fractal_utils::wind_base& wb) const noexcept final {
    if (!wb.float_type_matches<real_type>()) {
      return tl::make_unexpected(
          fmt::format("Can not set precision. Floating type mismatch."));
    }
    auto& wind = dynamic_cast<fu::center_wind<real_type>&>(wb);
    const auto seperater_location =
        hex.find_first_of(dynamic_precision_hex_seperator);
    if (seperater_location == std::string_view::npos) {
      return tl::make_unexpected(
          fmt::format("The hex string doesn't contain a separator (aka \'{}\')",
                      dynamic_precision_hex_seperator));
    }
    {
      const auto last_loc = hex.find_last_of(dynamic_precision_hex_seperator);
      if (last_loc != seperater_location) {
        return tl::make_unexpected(
            "The hex string contains multiple separators");
      }
    }
    std::vector<uint8_t> bin;
    auto fun_write_float = [hex, &wind, this](
                               size_t idx, std::span<const uint8_t> bin_single)
        -> tl::expected<void, std::string> {
      auto flt = fu::decode_float<real_type>(bin_single);
      if (!flt.has_value()) {
        return tl::make_unexpected(fmt::format(
            "Failed to decode the {}-th component of center.", idx));
      }
      wind.center[idx] = std::move(flt.value());
      return {};
    };
    {
      bin.resize(seperater_location * 2);
      auto bytes = fu::hex_2_bin({hex.data(), seperater_location}, bin);
      bin.resize(bytes.value());
      auto err = fun_write_float(0, bin);
      if (!err.has_value()) {
        return tl::make_unexpected(std::move(err.error()));
      }
    }
    {
      bin.resize(hex.size() * 2);
      auto bytes = fu::hex_2_bin(
          {hex.data() + seperater_location + 1, &*hex.end()}, bin);
      bin.resize(bytes.value());
      auto err = fun_write_float(1, bin);
      if (!err.has_value()) {
        return tl::make_unexpected(std::move(err.error()));
      }
    }
    return {};
  }
  static constexpr int min_precision = 50;
  [[nodiscard]] int suggested_precision_of(
      const fractal_utils::wind_base& _wind, int rows,
      int cols) const noexcept final {
    const auto& wind = dynamic_cast<const fu::center_wind<real_type>&>(_wind);
    return std::max<int>(min_precision,
                         (int)fu::required_precision_of(wind, rows, cols) + 4);
  }

  [[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
  clone_with_precision(const newton_equation_base& _src,
                       int p) const noexcept final {
    auto* srcp = dynamic_cast<const newton_equation_mpc*>(&_src);
    if (srcp == nullptr) {
      return tl::make_unexpected("The floating point type mismatch.");
    }

    auto ret = srcp->create_another_with_precision(p);
    if (ret.has_value()) {
      return std::move(ret.value());
    }
    return tl::make_unexpected(std::move(ret.error()));
  }
};

#endif

tl::expected<std::unique_ptr<object_creator>, std::string>
object_creator::create(fractal_utils::float_backend_lib backend, int precision,
                       gpu_backend gpu) noexcept {
  if (!is_valid_option(backend, precision, gpu)) {
    return tl::make_unexpected(
        fmt::format("Unsupported option: backend = {}, precision = {}.",
                    magic_enum::enum_name(backend), precision));
  }

  switch (gpu) {
    case gpu_backend::cuda: {
      object_creator* ret = nullptr;
      switch (precision) {
        case 1:
          ret = new cuda_object_creator<1>;
          break;
        case 2:
          ret = new cuda_object_creator<2>;
          break;
        default:
          return tl::make_unexpected(
              fmt::format("Unsupported option: backend = {}, precision = {}.",
                          magic_enum::enum_name(backend), precision));
      }
      return std::unique_ptr<object_creator>{ret};
    }
    case gpu_backend::opencl: {
      object_creator* ret = nullptr;
      switch (precision) {
        case 1:
          ret = new opencl_object_creator<1>;
          break;
        case 2:
          ret = new opencl_object_creator<2>;
          break;
        default:
          return tl::make_unexpected(
              fmt::format("Unsupported option: backend = {}, precision = {}.",
                          magic_enum::enum_name(backend), precision));
      }
      return std::unique_ptr<object_creator>{ret};
    }
    default:
      break;
  }

  switch (backend) {
#ifndef NF_USE_QUADMATH
    case fu::float_backend_lib::quadmath:
      return tl::make_unexpected(
          fmt::format("NewtonFractal is not built with quadmath."));
#else
    case fu::float_backend_lib::quadmath:
#endif
    case fu::float_backend_lib::standard:
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
    const newton_equation_base& neb, float_save_format fsf) const noexcept {
  return neb.to_json(fsf);
}

}  // namespace newton_fractal