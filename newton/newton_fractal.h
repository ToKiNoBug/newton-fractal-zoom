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

namespace newton_fractal {

struct meta_data {
  struct compute_objects {
    std::unique_ptr<object_creator> obj_creator{nullptr};
    std::unique_ptr<fractal_utils::wind_base> window{nullptr};
    std::unique_ptr<::nf::newton_equation_base> equation{nullptr};
  };
  struct non_compute_info {
    int num_points{0};
    fractal_utils::float_backend_lib backend{
        fractal_utils::float_backend_lib::unknown};
    int precision{0};
  };

  int rows{0};
  int cols{0};
  int iteration{0};
  std::variant<compute_objects, non_compute_info> compute_objs{
      compute_objects{}};

  [[nodiscard]] inline bool can_compute() const noexcept {
    return this->compute_objs.index() == 0;
  }

  meta_data() = default;
  meta_data(meta_data&&) = default;
  meta_data(const meta_data&) noexcept;

  meta_data& operator=(meta_data&&) noexcept = default;

  [[nodiscard]] auto obj_creator() noexcept {
    return std::get<0>(this->compute_objs).obj_creator.get();
  }
  [[nodiscard]] const object_creator* obj_creator() const noexcept {
    return std::get<0>(this->compute_objs).obj_creator.get();
  }

  [[nodiscard]] fractal_utils::wind_base* window() noexcept {
    return std::get<0>(this->compute_objs).window.get();
  }
  [[nodiscard]] const fractal_utils::wind_base* window() const noexcept {
    return std::get<0>(this->compute_objs).window.get();
  }
  [[nodiscard]] newton_equation_base* equation() noexcept {
    return std::get<0>(this->compute_objs).equation.get();
  }
  [[nodiscard]] const newton_equation_base* equation() const noexcept {
    return std::get<0>(this->compute_objs).equation.get();
  }

  [[nodiscard]] int num_points() const noexcept {
    if (this->compute_objs.index() == 0) {
      return this->equation()->order();
    }
    return std::get<1>(this->compute_objs).num_points;
  }
};

tl::expected<meta_data, std::string> load_metadata(
    const njson& nj, bool ignore_compute_objects) noexcept;
tl::expected<meta_data, std::string> load_metadata(
    std::string_view json, bool ignore_compute_objects) noexcept;
tl::expected<meta_data, std::string> load_metadata(
    std::istream& is, bool ignore_compute_objects) noexcept;

tl::expected<njson, std::string> save_metadata(const meta_data& m) noexcept;

};  // namespace newton_fractal

namespace fu = fractal_utils;
namespace nf = newton_fractal;

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_FRACTAL_H
