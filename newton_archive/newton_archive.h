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

  fu::unique_map map_is_number{0, 0, 1};

 public:
  auto &info() noexcept { return this->m_info; }
  const auto &info() const noexcept { return this->m_info; }
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_ARCHIVE_H
