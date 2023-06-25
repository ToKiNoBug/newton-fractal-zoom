#include "newton_fractal.h"
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <memory>
#include <iterator>

namespace newton_fractal {

meta_data::meta_data(const meta_data& src) noexcept
    : rows{src.rows}, cols{src.cols}, iteration{src.iteration} {
  if (src.compute_objs.index() == 1) {
    this->compute_objs = std::get<1>(src.compute_objs);
  }
  if (src.compute_objs.index() == 0) {
    const auto& src_cobjs = std::get<0>(src.compute_objs);
    this->compute_objs =
        compute_objects{.obj_creator = src_cobjs.obj_creator->copy(),
                        .window{src_cobjs.window->create_another()},
                        .equation = src_cobjs.equation->copy()};
    src_cobjs.window->copy_to(this->window());
  } else {
    this->compute_objs = std::get<1>(src.compute_objs);
  }
}

tl::expected<meta_data, std::string> load_metadata(
    std::string_view json, bool ignore_compute_objects) noexcept {
  njson nj;
  try {
    nj = njson::parse(json, nullptr, true, true);
  } catch (std::exception& e) {
    return tl::make_unexpected(
        fmt::format("Failed to parse json. Detail: {}.", e.what()));
  }

  return load_metadata(nj, ignore_compute_objects);
}

tl::expected<meta_data, std::string> load_metadata(
    std::istream& is, bool ignore_compute_objects) noexcept {
  njson nj;
  try {
    nj = njson::parse(is, nullptr, true, true);
  } catch (std::exception& e) {
    return tl::make_unexpected(
        fmt::format("Failed to parse json. Detail: {}.", e.what()));
  }

  return load_metadata(nj, ignore_compute_objects);
}

tl::expected<meta_data, std::string> load_metadata(
    const njson& nj, bool ignore_compute_objects) noexcept {
  meta_data ret;

  try {
    std::string backend_str = nj.at("backend");
    const auto backend =
        magic_enum::enum_cast<fu::float_backend_lib>(backend_str);
    if (!backend.has_value()) {
      return tl::make_unexpected(
          fmt::format("Unknown backend type \"{}\"", backend_str));
    }

    const int precision = nj.at("precision");

    ret.rows = nj.at("rows");
    ret.cols = nj.at("cols");
    ret.iteration = nj.at("iteration");

    if (ret.rows <= 1 || ret.cols <= 1) {
      return tl::make_unexpected(fmt::format(
          "Invalid value for rows({}) and cols({}).", ret.rows, ret.cols));
    }
    if (ret.iteration <= 1) {
      return tl::make_unexpected(fmt::format(
          "The value for iteration({}) is too small.", ret.iteration));
    }

    if (ignore_compute_objects) {
      ret.compute_objs = meta_data::non_compute_info{
          .num_points = (int)nj.at("points").size()};
      return ret;
    }

    meta_data::compute_objects cobj;
    {
      auto objc = object_creator::create(backend.value(), precision);
      if (!objc.has_value()) {
        return tl::make_unexpected(fmt::format(
            "Failed to create object_creator. Detail: {}", objc.error()));
      }

      cobj.obj_creator = std::move(objc.value());
    }

    {
      auto wind_e = cobj.obj_creator->create_window(nj.at("window"));
      if (!wind_e.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to create window. Detail: {}", wind_e.error()));
      }
      cobj.window = std::move(wind_e.value());
    }

    {
      auto eq_e = cobj.obj_creator->create_equation(nj.at("points"));
      if (!eq_e.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to create equation. Detail: {}", eq_e.error()));
      }
      cobj.equation = std::move(eq_e.value());
    }

    ret.compute_objs = std::move(cobj);

  } catch (std::exception& e) {
    return tl::make_unexpected(fmt::format(
        "Exception occurred when parsing json. Detail: {}", e.what()));
  }

  return ret;
}

tl::expected<njson, std::string> save_metadata(const meta_data& m) noexcept {
  if (std::get<0>(m.compute_objs).obj_creator == nullptr) {
    return tl::make_unexpected("The object creator pointer is null.");
  }
  if (std::get<0>(m.compute_objs).equation == nullptr) {
    return tl::make_unexpected("The newton equation pointer is null.");
  }
  if (std::get<0>(m.compute_objs).window == nullptr) {
    return tl::make_unexpected("The window pointer is null.");
  }

  njson ret;
  ret.emplace("backend",
              magic_enum::enum_name(
                  std::get<0>(m.compute_objs).obj_creator->backend_lib()));
  ret.emplace("precision",
              std::get<0>(m.compute_objs).obj_creator->precision());

  {
    auto wind =
        std::get<0>(m.compute_objs)
            .obj_creator->save_window(*std::get<0>(m.compute_objs).window);
    if (!wind.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to save window because {}", wind.error()));
    }
    ret.emplace("window", std::move(wind.value()));
  }
  {
    auto points =
        std::get<0>(m.compute_objs)
            .obj_creator->save_equation(*std::get<0>(m.compute_objs).equation);
    ret.emplace("points", std::move(points));
  }

  ret.emplace("rows", m.rows);
  ret.emplace("cols", m.cols);
  ret.emplace("iteration", m.iteration);

  return ret;
}
}  // namespace newton_fractal
