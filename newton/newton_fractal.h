//
// Created by David on 2023/6/17.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_FRACTAL_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_FRACTAL_H

#include <core_utils.h>
#include "multiprecision_utils.h"
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
#include <memory>
#include "newton_equation.hpp"
#include "object_creator.h"

#ifdef NEWTON_FRACTAL_MPC_SUPPORT
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>

#include "mpc_support.h"

#endif

namespace fu = fractal_utils;

#define NEWTONFRACTAL_FLOAT_TYPE_LIST                           \
  fu::float_by_precision_t<1>, fu::float_by_precision_t<2>,     \
      fu::float_by_precision_t<4>, fu::float_by_precision_t<8>, \
      fu::float_by_precision_t<16>

namespace newton_fractal {
using number_variant_t = std::variant<NEWTONFRACTAL_FLOAT_TYPE_LIST>;

tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_equation(fractal_utils::float_backend_lib backend,
                int precision) noexcept;

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_FRACTAL_H
