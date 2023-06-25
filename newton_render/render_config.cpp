#include "render_config.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <magic_enum.hpp>

using njson = nlohmann::json;

namespace newton_fractal {

tl::expected<render_config::render_method::color_value_mapping, std::string>
parse_color_value_mapping(const njson &nj) noexcept {
  render_config::render_method::color_value_mapping cvm{};
  try {
    const auto &range_arr = nj.at("range");
    if (range_arr.size() != 2) {
      return tl::make_unexpected(fmt::format(
          "range should be of size 2, but actually {}", range_arr.size()));
    }
    cvm.range[0] = range_arr[0];
    cvm.range[1] = range_arr[1];

    std::string src_str = nj.at("source");
    auto source_opt =
        magic_enum::enum_cast<render_config::mapping_source>(src_str);
    if (!source_opt.has_value()) {
      return tl::make_unexpected(
          fmt::format("Invalid render source option named \"{}\"", src_str));
    }
    cvm.src = source_opt.value();

  } catch (std::exception &e) {
    return tl::make_unexpected(fmt::format(
        "Exception occurred while parsing "
        "render_config::render_method::color_value_mapping, detail: {}",
        e.what()));
  }

  return cvm;
}

tl::expected<render_config::render_method, std::string> parse_render_method(
    const njson &nj) noexcept {
  render_config::render_method ret{};
  try {
    {
      auto mapping = parse_color_value_mapping(nj.at("hue"));
      if (!mapping.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to parse hue, detail: {}", mapping.error()));
      }
      ret.hue = mapping.value();
    }
    {
      auto mapping = parse_color_value_mapping(nj.at("saturation"));
      if (!mapping.has_value()) {
        return tl::make_unexpected(fmt::format(
            "Failed to parse saturation, detail: {}", mapping.error()));
      }
      ret.saturation = mapping.value();
    }
    {
      auto mapping = parse_color_value_mapping(nj.at("value"));
      if (!mapping.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to parse value, detail: {}", mapping.error()));
      }
      ret.value = mapping.value();
    }
  } catch (std::exception &e) {
    return tl::make_unexpected(
        fmt::format("Exception occurred while parsing "
                    "render_config::render_method, detail: {}",
                    e.what()));
  }
  return ret;
}

tl::expected<render_config, std::string> parse_render_config(
    const njson &nj) noexcept {
  render_config ret;
  try {
    const auto &color_nan_arr = nj.at("color_for_nan");
    if (color_nan_arr.size() != 3) {
      return tl::make_unexpected(
          fmt::format("color_for_nan should be an array with size = 3, but "
                      "actual size is {}.",
                      color_nan_arr.size()));
    }

    for (int c = 0; c < 3; c++) {
      float val = color_nan_arr[c];
      if (val < 0 || val > 1) {
        return tl::make_unexpected(
            fmt::format("The rgb value {} is not in range [0,1]", val));
      }
      ret.color_for_nan.value[c] = uint8_t(val * 255);
    }

    auto &methods_arr = nj.at("point_methods");
    ret.methods.reserve(methods_arr.size());
    for (size_t i = 0; i < methods_arr.size(); i++) {
      const auto &mj = methods_arr[i];

      auto method = parse_render_method(mj);
      if (!method.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to parse render method at index {}, detail: {}",
                        i, method.error()));
      }
      ret.methods.emplace_back(method.value());
    }
  } catch (std::exception &e) {
    return tl::make_unexpected(fmt::format(
        "Exception occurred while parsing render config, detail: {}",
        e.what()));
  }

  return ret;
}

tl::expected<render_config, std::string> load_render_config(
    std::istream &is) noexcept {
  njson nj;
  try {
    nj = njson::parse(is, nullptr, true, true);

    auto ret = parse_render_config(nj);
    return ret;
  } catch (std::exception &e) {
    return tl::make_unexpected(fmt::format(
        "Exception occurred while parsing json, detail: {}", e.what()));
  }
}

tl::expected<render_config, std::string> load_render_config(
    std::string_view filename) noexcept {
  std::ifstream ifs{filename.data()};
  return load_render_config(ifs);
}

}  // namespace newton_fractal