#include "newton_fractal.h"
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <memory>

namespace nf = newton_fractal;

tl::expected<std::unique_ptr<nf::newton_equation_base>, std::string>
nf::create_equation(fractal_utils::float_backend_lib backend,
                    int precision) noexcept {
  switch (backend) {
    case fu::float_backend_lib::standard: {
      switch (precision) {
        case 1:
          return std::make_unique<nf::equation_fixed_prec<1>>();
        case 2:
          return std::make_unique<nf::equation_fixed_prec<2>>();
        default:
          return tl::make_unexpected(fmt::format(
              "Precision {} is invalid for cpp standard", precision));
      }
    }
    case fu::float_backend_lib::quadmath:
    case fu::float_backend_lib::boost: {
      switch (precision) {
        case 4:
          return std::make_unique<nf::equation_fixed_prec<4>>();
        case 8:
          return std::make_unique<nf::equation_fixed_prec<8>>();
        case 16:
          return std::make_unique<nf::equation_fixed_prec<16>>();
        default:
          return tl::make_unexpected(
              fmt::format("Precision {} is invalid or unsupported for {}",
                          precision, magic_enum::enum_name(backend)));
      }

      case fu::float_backend_lib::mpfr: {
        if (precision <= 0) {
          return tl::make_unexpected(
              "Precision for mpfr/mpc must be positive number.");
        }

        return std::make_unique<newton_equation_mpc>(precision);
      }
    }
    default:
      return tl::make_unexpected(fmt::format("Unsupported backend type {}",
                                             magic_enum::enum_name(backend)));
  }
}