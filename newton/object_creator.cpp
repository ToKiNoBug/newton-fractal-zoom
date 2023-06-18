//
// Created by joseph on 6/19/23.
//

#include "object_creator.h"
#include <magic_enum.hpp>

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
    }
#else
      return tl::make_unexpected("mpc support is disabled.");
#endif

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

tl::expected<std::unique_ptr<object_creator>, std::string>
object_creator::create(fractal_utils::float_backend_lib backend,
                       int precision) noexcept {}

}  // namespace newton_fractal