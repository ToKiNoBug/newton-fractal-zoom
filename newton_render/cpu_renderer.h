//
// Created by joseph on 6/29/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_CPU_RENDERER_H
#define NEWTON_FRACTAL_ZOOM_CPU_RENDERER_H

#include "render_config.h"
#include <unique_map.h>
#include <tl/expected.hpp>
#include <string>
#include <variant>
#include <optional>

namespace newton_fractal {

namespace internal {

void set_data(fractal_utils::constant_view src,
              std::optional<std::variant<fractal_utils::unique_map,
                                         fractal_utils::constant_view>>& dst,
              bool deep_copy) noexcept;

[[nodiscard]] fractal_utils::constant_view get_map(
    const std::optional<
        std::variant<fractal_utils::unique_map, fractal_utils::constant_view>>&
        src) noexcept;

[[nodiscard]] uint8_t compute_max_nearest_index(
    fractal_utils::constant_view has_value,
    fractal_utils::constant_view nearest_index) noexcept;
}  // namespace internal

class cpu_renderer {
 private:
  fractal_utils::unique_map m_map_norm_arg;
  std::optional<
      std::variant<fractal_utils::unique_map, fractal_utils::constant_view>>
      m_has_value{std::nullopt};
  std::optional<
      std::variant<fractal_utils::unique_map, fractal_utils::constant_view>>
      m_nearest_idx{std::nullopt};
  std::optional<
      std::variant<fractal_utils::unique_map, fractal_utils::constant_view>>
      m_complex_difference{std::nullopt};

  std::optional<uint8_t> m_nearest_idx_max{std::nullopt};

  void compute_norm_arg() & noexcept;
  void compute_nearest_idx_max() & noexcept;

  [[nodiscard]] auto rows() const noexcept {
    return std::visit([](const auto& val) { return val.rows(); },
                      this->m_has_value.value());
  }
  [[nodiscard]] auto cols() const noexcept {
    return std::visit([](const auto& val) { return val.cols(); },
                      this->m_has_value.value());
  }

 public:
  cpu_renderer() = default;
  cpu_renderer(const cpu_renderer&) = delete;
  cpu_renderer(cpu_renderer&&) = default;

  cpu_renderer& operator=(const cpu_renderer&) = delete;
  cpu_renderer& operator=(cpu_renderer&&) = delete;

  void set_data(fractal_utils::constant_view has_value,
                fractal_utils::constant_view map_nearest_idx,
                fractal_utils::constant_view map_complex_difference,
                bool deep_copy) & noexcept;
  void set_data(fractal_utils::unique_map&& has_value,
                fractal_utils::unique_map&& map_nearest_idx,
                fractal_utils::unique_map&& map_complex_difference) & noexcept;

  void reset() & noexcept;

  void render(const render_config& config,
              fractal_utils::constant_view map_has_value,
              fractal_utils::constant_view map_nearest_idx,
              fractal_utils::constant_view map_complex_difference,
              fractal_utils::map_view image_u8c3, int skip_rows,
              int skip_cols) & noexcept;

  [[nodiscard]] tl::expected<void, std::string> render(
      const render_config& config, fractal_utils::map_view image_u8c3,
      int skip_rows, int skip_cols) const noexcept;

  //  [[nodiscard]] inline auto rows() const noexcept {
  //    return this->m_map_norm_arg.rows();
  //  }
  //  [[nodiscard]] inline auto cols() const noexcept {
  //    return this->m_map_norm_arg.cols();
  //  }
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_CPU_RENDERER_H
