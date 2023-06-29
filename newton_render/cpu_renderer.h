//
// Created by joseph on 6/29/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_CPU_RENDERER_H
#define NEWTON_FRACTAL_ZOOM_CPU_RENDERER_H

#include "render_config.h"
#include <unique_map.h>
#include <tl/expected.hpp>
#include <string>

namespace newton_fractal {

class cpu_renderer {
 private:
  fractal_utils::unique_map m_map_norm_arg;

 public:
  cpu_renderer() = default;

  void render(const render_config& config,
              fractal_utils::constant_view map_has_value,
              fractal_utils::constant_view map_nearest_idx,
              fractal_utils::constant_view map_complex_difference,
              fractal_utils::map_view image_u8c3, int skip_rows,
              int skip_cols) & noexcept;

  //  [[nodiscard]] inline auto rows() const noexcept {
  //    return this->m_map_norm_arg.rows();
  //  }
  //  [[nodiscard]] inline auto cols() const noexcept {
  //    return this->m_map_norm_arg.cols();
  //  }
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_CPU_RENDERER_H
