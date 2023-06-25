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

#ifdef __CUDACC__
#define NF_HOST_DEVICE_FUN __host__ __device__
#else
#define NF_HOST_DEVICE_FUN
#endif

namespace newton_fractal {

struct render_config {
  enum class mapping_source : uint8_t { magnitude, angle };

  struct render_method {
    struct color_value_mapping {
      float range[2];
      mapping_source src;

      [[nodiscard]] NF_HOST_DEVICE_FUN inline float map(
          float mag_normalized, float arg_normalized) const noexcept {
        const float before_mapping = (this->src == mapping_source::magnitude)
                                         ? (mag_normalized)
                                         : (arg_normalized);
        return before_mapping * (this->range[1] - this->range[0]) +
               this->range[0];
      }
    };
    color_value_mapping hue;
    color_value_mapping saturation;
    color_value_mapping value;

    NF_HOST_DEVICE_FUN inline void map_color(float mag_normalized,
                                             float arg_normalized, float &H,
                                             float &S,
                                             float &V) const noexcept {
      H = this->hue.map(mag_normalized, arg_normalized);
      S = this->saturation.map(mag_normalized, arg_normalized);
      V = this->value.map(mag_normalized, arg_normalized);
    }
  };

  std::vector<render_method> methods;
  fractal_utils::pixel_RGB color_for_nan{0xFF000000};
};

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

  [[nodiscard]] virtual bool ok() const noexcept = 0;
  [[nodiscard]] virtual int error_code() const noexcept = 0;

  [[nodiscard]] virtual fractal_utils::pixel_RGB color_for_nan()
      const noexcept = 0;

  [[nodiscard]] static tl::expected<
      std::unique_ptr<render_config_gpu_interface>, std::string>
  create() noexcept;
};

class gpu_interface {
 public:
  virtual ~gpu_interface() = default;

  [[nodiscard]] virtual int rows() const noexcept = 0;
  [[nodiscard]] virtual int cols() const noexcept = 0;
  [[nodiscard]] inline int size() const noexcept {
    return this->rows() * this->cols();
  }

  [[nodiscard]] virtual bool ok() const noexcept = 0;

  [[nodiscard]] virtual int error_code() const noexcept = 0;

  [[nodiscard]] virtual tl::expected<void, std::string> set_has_value(
      std::span<const bool> src) & noexcept = 0;
  [[nodiscard]] virtual tl::expected<void, std::string> set_nearest_index(
      std::span<const uint8_t> src) & noexcept = 0;
  [[nodiscard]] virtual tl::expected<void, std::string> set_complex_difference(
      std::span<const std::complex<double>> src) & noexcept = 0;

  [[nodiscard]] static tl::expected<std::unique_ptr<gpu_interface>, std::string>
  create(int rows, int cols) noexcept;

  [[nodiscard]] virtual tl::expected<void, std::string> run(
      const render_config_gpu_interface &config, int skip_rows, int skip_cols,
      bool sync) & noexcept = 0;

  [[nodiscard]] virtual tl::expected<void, std::string> get_pixels(
      fractal_utils::map_view image_u8c3) & noexcept = 0;
};

NF_HOST_DEVICE_FUN inline fractal_utils::pixel_RGB render(
    const render_config::render_method *method_ptr,
    fractal_utils::pixel_RGB color_for_nan, bool has_value, int nearest_idx,
    float mag_normalized, float arg_normalized) noexcept {
  if (!has_value) {
    return color_for_nan;
  }

  float h, s, v;
  method_ptr[nearest_idx].map_color(mag_normalized, arg_normalized, h, s, v);

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
  assert(methods.size() > nearest_idx);
  assert(nearest_idx >= 0);
  assert(mag_normalized >= 0 && mag_normalized <= 1);
  assert(arg_normalized >= 0 && arg_normalized <= 1);
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

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H
