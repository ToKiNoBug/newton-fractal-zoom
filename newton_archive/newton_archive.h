//
// Created by joseph on 6/18/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_ARCHIVE_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_ARCHIVE_H

#include <newton_fractal.h>
#include <core_utils.h>

namespace newton_fractal {

class newton_archive {
 private:
  meta_data m_info;

  fu::unique_map m_map_has_result{0, 0, sizeof(bool)};
  fu::unique_map m_map_nearest_point_idx{0, 0, sizeof(uint8_t)};
  fu::unique_map m_map_complex_difference{0, 0, sizeof(std::complex<double>)};

 public:
  [[nodiscard]] auto &info() noexcept { return this->m_info; }
  [[nodiscard]] const auto &info() const noexcept { return this->m_info; }

  auto &map_has_result() noexcept { return this->m_map_has_result; }
  [[nodiscard]] const auto &map_has_result() const noexcept {
    return this->m_map_has_result;
  }

  [[nodiscard]] auto &map_nearest_point_idx() noexcept {
    return this->m_map_nearest_point_idx;
  }
  [[nodiscard]] const auto &map_nearest_point_idx() const noexcept {
    return this->m_map_nearest_point_idx;
  }

  [[nodiscard]] auto &map_complex_difference() noexcept {
    return this->m_map_complex_difference;
  }
  [[nodiscard]] const auto &map_complex_difference() const noexcept {
    return this->m_map_complex_difference;
  }
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_ARCHIVE_H
