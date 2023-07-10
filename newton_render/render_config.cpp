#include "render_config.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include "newton_render.h"

using njson = nlohmann::json;

namespace newton_fractal {

tl::expected<render_config::render_method::color_value_mapping, std::string>
parse_color_value_mapping(const njson &nj) noexcept {
  render_config::render_method::color_value_mapping cvm{};
  bool is_range_single{false};
  try {
    const auto &range = nj.at("range");

    if (range.is_array()) {
      if (range.size() != 2) {
        return tl::make_unexpected(fmt::format(
            "range should be of size 2, but actually {}", range.size()));
      }
      cvm.range[0] = range[0];
      cvm.range[1] = range[1];
    } else {
      is_range_single = true;
      cvm.range[0] = range;
      cvm.range[1] = range;
    }
    std::string src_str;
    if (!nj.contains("source")) {
      if (!is_range_single) {
        return tl::make_unexpected(
            "The range is not fixed at a single value, so source can not be "
            "omitted.");
      }
      src_str = "magnitude";
    } else {
      src_str = nj.at("source");
    }
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

tl::expected<void, std::string> check_mapping(
    const render_config::render_method::color_value_mapping &m,
    double lower_bound, double upper_bound) noexcept {
  for (size_t idx = 0; idx < 2; idx++) {
    if (m.range[idx] < lower_bound || m.range[idx] > upper_bound) {
      return tl::make_unexpected(fmt::format(
          "The range should be in range [{}, {}], but met {} at index {}.",
          lower_bound, upper_bound, m.range[idx], idx));
    }
  }
  return {};
}

tl::expected<void, std::string> check_hue(
    const render_config::render_method::color_value_mapping &m) {
  return check_mapping(m, 0, std::nextafter<double>(360, -1));
}
tl::expected<void, std::string> check_saturation_value(
    const render_config::render_method::color_value_mapping &m) {
  return check_mapping(m, 0, 1);
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
      auto err = check_hue(ret.hue);
      if (!err) {
        return tl::make_unexpected(
            fmt::format("The range of hue is invalid: {}", err.error()));
      }
    }
    {
      auto mapping = parse_color_value_mapping(nj.at("saturation"));
      if (!mapping.has_value()) {
        return tl::make_unexpected(fmt::format(
            "Failed to parse saturation, detail: {}", mapping.error()));
      }
      ret.saturation = mapping.value();
      auto err = check_saturation_value(ret.saturation);
      if (!err) {
        fmt::format("The range of saturation is invalid: {}", err.error());
      }
    }
    {
      auto mapping = parse_color_value_mapping(nj.at("value"));
      if (!mapping.has_value()) {
        return tl::make_unexpected(
            fmt::format("Failed to parse value, detail: {}", mapping.error()));
      }
      ret.value = mapping.value();
      auto err = check_saturation_value(ret.value);
      if (!err) {
        fmt::format("The range of value is invalid: {}", err.error());
      }
    }
  } catch (std::exception &e) {
    return tl::make_unexpected(
        fmt::format("Exception occurred while parsing "
                    "render_config::render_method, detail: {}",
                    e.what()));
  }
  return ret;
}

tl::expected<render_config, std::string> load_render_config(
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

    auto ret = load_render_config(nj);
    return ret;
  } catch (std::exception &e) {
    return tl::make_unexpected(fmt::format(
        "Exception occurred while parsing json, detail: {}", e.what()));
  }
}

tl::expected<render_config, std::string> load_render_config_from_file(
    std::string_view filename) noexcept {
  std::ifstream ifs{filename.data()};
  if (!ifs) {
    return tl::make_unexpected(
        fmt::format("Failed to open file \"{}\"", filename));
  }
  return load_render_config(ifs);
}

njson save_mapping(
    const render_config::render_method::color_value_mapping &cvm) noexcept {
  njson ret;
  if (cvm.range[0] == cvm.range[1]) {
    ret.emplace("range", cvm.range[0]);
  } else {
    njson::array_t arr;
    arr.resize(2);
    arr[0] = cvm.range[0];
    arr[1] = cvm.range[1];
    ret.emplace("range", std::move(arr));
  }

  ret.emplace("source", magic_enum::enum_name(cvm.src));
  return ret;
}

njson save_render_method(const render_config::render_method &rm) noexcept {
  njson ret;
  ret.emplace("hue", save_mapping(rm.hue));
  ret.emplace("saturation", save_mapping(rm.saturation));
  ret.emplace("value", save_mapping(rm.value));
  return ret;
}

njson save_render_config(const render_config &nc) noexcept {
  njson ret;
  {
    njson::array_t methods;
    methods.reserve(nc.methods.size());
    for (const auto &m : nc.methods) {
      methods.emplace_back(save_render_method(m));
    }
    ret.emplace("point_methods", std::move(methods));
  }
  {
    njson::array_t color_for_nan;
    color_for_nan.resize(3);
    for (size_t idx = 0; idx < 3; idx++) {
      color_for_nan[idx] = nc.color_for_nan.value[idx];
    }
    ret.emplace("color_for_nan", std::move(color_for_nan));
  }
  return ret;
}

void serialize_render_config(const render_config &rc,
                             std::string &ret) noexcept {
  njson json = save_render_config(rc);
  ret = json.dump(2);
}

std::string serialize_render_config(const render_config &rc) noexcept {
  std::string ret;
  serialize_render_config(rc, ret);
  return ret;
}

}  // namespace newton_fractal