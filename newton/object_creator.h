//
// Created by joseph on 6/19/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_OBJECT_CREATOR_H
#define NEWTON_FRACTAL_ZOOM_OBJECT_CREATOR_H

#include <multiprecision_utils.h>
#include "newton_equation.hpp"
#include <tl/expected.hpp>

namespace newton_fractal {

tl::expected<void, std::string> is_valid_option(
    fractal_utils::float_backend_lib backend, int precision) noexcept;

class object_creator {
 public:
  virtual ~object_creator() = default;

  static tl::expected<std::unique_ptr<object_creator>, std::string> create(
      fractal_utils::float_backend_lib backend, int precision) noexcept;

  [[nodiscard]] virtual fractal_utils::float_backend_lib backend_lib()
      const noexcept = 0;
  [[nodiscard]] virtual int precision() const noexcept = 0;

  [[nodiscard]] virtual tl::expected<std::unique_ptr<fractal_utils::wind_base>,
                                     std::string>
  create_window(const njson&) const noexcept = 0;

  [[nodiscard]] virtual tl::expected<std::unique_ptr<fractal_utils::wind_base>,
                                     std::string>
  create_equation(const njson&) const noexcept = 0;
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_OBJECT_CREATOR_H
