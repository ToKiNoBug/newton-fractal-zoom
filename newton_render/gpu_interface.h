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

#ifdef __CUDA__
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
  uint32_t color_for_nan{0xFF000000};
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
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H
