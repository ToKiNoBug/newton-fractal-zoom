//
// Created by David on 2023/6/21.
//

#ifndef NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H
#define NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H

#include <memory>
#include <complex>
#include <tl/expected.hpp>
#include <string>
#include <optional>
#include <span>
#include <array>
#include <cassert>
#include <color_cvt.hpp>
#include <fractal_colors.h>
#include <unique_map.h>
#include "render_config.h"
#include "NF_cuda_macros.hpp"

namespace fu = fractal_utils;

namespace newton_fractal {

class render_config_gpu_interface {
 public:
  virtual ~render_config_gpu_interface() = default;

  [[nodiscard]] virtual tl::expected<void, std::string> set_config(
      const render_config &rc) & noexcept = 0;
  [[nodiscard]] virtual tl::expected<render_config, std::string> config()
      const noexcept = 0;

  [[nodiscard]] virtual const render_config::render_method *method_ptr()
      const noexcept = 0;
  [[nodiscard]] virtual int num_methods() const noexcept = 0;

  //[[nodiscard]] virtual bool ok() const noexcept = 0;
  //[[nodiscard]] virtual int error_code() const noexcept = 0;

  [[nodiscard]] virtual fractal_utils::pixel_RGB color_for_nan()
      const noexcept = 0;

  [[nodiscard]] static tl::expected<
      std::unique_ptr<render_config_gpu_interface>, std::string>
  create() noexcept;
};

class gpu_render {
 public:
  [[nodiscard]] static tl::expected<std::unique_ptr<gpu_render>, std::string>
  create(int rows, int cols) noexcept;
  virtual ~gpu_render() = default;

  //[[nodiscard]] virtual bool ok() const noexcept = 0;

  //[[nodiscard]] virtual int error_code() const noexcept = 0;

  [[nodiscard]] virtual tl::expected<void, std::string> set_data(
      fractal_utils::constant_view has_value,
      fractal_utils::constant_view map_nearest_idx,
      fractal_utils::constant_view map_complex_difference,
      bool deep_copy) & noexcept = 0;

  [[nodiscard]] virtual tl::expected<void, std::string> render(
      const render_config_gpu_interface &config, fu::constant_view has_value,
      fu::constant_view nearest_index, fu::constant_view complex_difference,
      fu::map_view image_u8c3, int skip_rows, int skip_cols) & noexcept = 0;

  [[nodiscard]] virtual tl::expected<fractal_utils::constant_view, std::string>
  render(const render_config_gpu_interface &config, int skip_rows,
         int skip_cols) & noexcept = 0;

  [[nodiscard]] tl::expected<void, std::string> render(
      const render_config_gpu_interface &config,
      fractal_utils::map_view image_u8c3, int skip_rows,
      int skip_cols) & noexcept;
};

NF_HOST_DEVICE_FUN inline fractal_utils::pixel_RGB render(
    const render_config::render_method *method_ptr,
    fractal_utils::pixel_RGB color_for_nan, bool has_value, int nearest_idx,
    float mag_normalized, float arg_normalized) noexcept {
  if (!has_value) {
    return color_for_nan;
  }

  static_assert(sizeof(render_config::render_method) == 36);

  float h, s, v;
  method_ptr[nearest_idx].map_color(mag_normalized, arg_normalized, h, s, v);
  assert(nearest_idx < 255);

  //  if (nearest_idx < 0 || nearest_idx >= 3) {
  //    printf("nearest_idx=%i maps to h = %f, s = %f, v = %f.\n",
  //    int(nearest_idx),
  //           h, s, v);
  //  }

  float rgb[3];
  fractal_utils::hsv_to_rgb(h, s, v, rgb[0], rgb[1], rgb[2]);
  fractal_utils::pixel_RGB ret{};
  for (int i = 0; i < 3; i++) {
    assert(rgb[i] >= 0);
    assert(rgb[i] <= 1);
    ret.value[i] = uint8_t(255.0f * rgb[i]);
  }
  return ret;
}

inline fractal_utils::pixel_RGB render_cpu(
    std::span<const render_config::render_method> methods,
    fractal_utils::pixel_RGB color_for_nan, bool has_value, int nearest_idx,
    float mag_normalized, float arg_normalized) noexcept {
  if (has_value) {
    assert(methods.size() > nearest_idx);
    assert(nearest_idx >= 0);
    assert(mag_normalized >= 0 && mag_normalized <= 1);
    assert(arg_normalized >= 0 && arg_normalized <= 1);
  }
  return render(methods.data(), color_for_nan, has_value, nearest_idx,
                mag_normalized, arg_normalized);
}

inline fractal_utils::pixel_RGB render_cpu(const render_config &cfg,
                                           bool has_value, int nearest_idx,
                                           float mag_normalized,
                                           float arg_normalized) noexcept {
  return render_cpu(cfg.methods, cfg.color_for_nan, has_value, nearest_idx,
                    mag_normalized, arg_normalized);
}

struct normalize_option {
  // using min_max_type = std::conditional_t<atomic, std::atomic<double>,
  // double>;
  double min;
  double max;

  [[nodiscard]] NF_HOST_DEVICE_FUN inline double normalize(
      double src) const noexcept {
    assert(src >= this->min);
    assert(src <= this->max);
    assert(this->max != this->min);
    return (src - this->min) / (this->max - this->min);
  }

  NF_HOST_DEVICE_FUN inline void add_data(double d) & noexcept {
    this->min = std::min(this->min, d);
    this->max = std::max(this->max, d);
  }
};
}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H
