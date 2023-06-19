//
// Created by David on 2023/6/17.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_FRACTAL_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_FRACTAL_H

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
#include <memory>
#include "newton_equation_base.h"
#include "object_creator.h"

#ifdef NEWTON_FRACTAL_MPC_SUPPORT
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>
#endif

namespace fu = fractal_utils;
namespace nf = newton_fractal;

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_FRACTAL_H
