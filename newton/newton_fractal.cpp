#include "newton_fractal.h"
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <memory>
#include <iterator>

namespace newton_fractal {

tl::expected<meta_data, std::string> load_metadata(
    std::string_view json) noexcept {
  njson nj;
  try {
    nj = njson::parse(json, nullptr, true, true);
  } catch (std::exception& e) {
    return tl::make_unexpected(
        fmt::format("Failed to parse json. Detail: {}.", e.what()));
  }

  return load_metadata(nj);
}

tl::expected<meta_data, std::string> load_metadata(std::istream& is) noexcept {
  njson nj;
  try {
    nj = njson::parse(is, nullptr, true, true);
  } catch (std::exception& e) {
    return tl::make_unexpected(
        fmt::format("Failed to parse json. Detail: {}.", e.what()));
  }

  return load_metadata(nj);
}

tl::expected<meta_data, std::string> load_metadata(const njson& nj) noexcept {
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

    {
      auto objc = object_creator::create(backend.value(), precision);
      if (!objc.has_value()) {
        return tl::make_unexpected(fmt::format(
            "Failed to create object_creator. Detail: {}", objc.error()));
      }

      ret.obj_creator = std::move(objc.value());
    }

    {
      auto wind_e = ret.obj_creator->create_window(nj.at("window"));
      if (!wind_e.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to create window. Detail: {}", wind_e.error()));
      }
      ret.window = std::move(wind_e.value());
    }

    {
      auto eq_e = ret.obj_creator->create_equation(nj.at("points"));
      if (!eq_e.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to create equation. Detail: {}", eq_e.error()));
      }
      ret.equation = std::move(eq_e.value());
    }

  } catch (std::exception& e) {
    return tl::make_unexpected(fmt::format(
        "Exception occurred when parsing json. Detail: {}", e.what()));
  }

  return ret;
}

tl::expected<njson, std::string> save_metadata(const meta_data& m) noexcept {
  if (m.obj_creator == nullptr) {
    return tl::make_unexpected("The object creator pointer is null.");
  }
  if (m.equation == nullptr) {
    return tl::make_unexpected("The newton equation pointer is null.");
  }
  if (m.window == nullptr) {
    return tl::make_unexpected("The window pointer is null.");
  }

  njson ret;
  ret.emplace("backend", magic_enum::enum_name(m.obj_creator->backend_lib()));
  ret.emplace("precision", m.obj_creator->precision());

  {
    auto wind = m.obj_creator->save_window(*m.window);
    if (!wind.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to save window because {}", wind.error()));
    }
    ret.emplace("window", std::move(wind.value()));
  }
  {
    auto points = m.obj_creator->save_equation(*m.equation);
    ret.emplace("points", std::move(points));
  }

  ret.emplace("rows", m.rows);
  ret.emplace("cols", m.cols);
  ret.emplace("iteration", m.iteration);

  return ret;
}
}  // namespace newton_fractal
