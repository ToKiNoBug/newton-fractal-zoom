//
// Created by joseph on 6/25/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_RENDER_CONFIG_H
#define NEWTON_FRACTAL_ZOOM_RENDER_CONFIG_H

#include <tl/expected.hpp>
#include <vector>
#include <fractal_colors.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <istream>
#include "NF_cuda_macros.hpp"

/*
namespace nlohmann {
class json;
}
using njson = nlohmann::json;
*/

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

tl::expected<render_config, std::string> load_render_config(
    std::istream &src) noexcept;
tl::expected<render_config, std::string> load_render_config(
    std::string_view filename) noexcept;

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_RENDER_CONFIG_H
