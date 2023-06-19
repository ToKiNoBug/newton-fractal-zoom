//
// Created by joseph on 6/19/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_OBJECT_CREATOR_H
#define NEWTON_FRACTAL_ZOOM_OBJECT_CREATOR_H

#include <multiprecision_utils.h>
#include "newton_equation_base.h"
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

  [[nodiscard]] virtual bool is_fixed_precision() const noexcept = 0;

  [[nodiscard]] virtual tl::expected<std::unique_ptr<fractal_utils::wind_base>,
                                     std::string>
  create_window(const njson&) const noexcept = 0;

  [[nodiscard]] virtual tl::expected<std::unique_ptr<newton_equation_base>,
                                     std::string>
  create_equation(const njson&) const noexcept = 0;

  [[nodiscard]] virtual tl::expected<njson, std::string> save_window(
      const fractal_utils::wind_base& wb) const noexcept = 0;

  [[nodiscard]] virtual njson::array_t save_equation(
      const newton_equation_base&) const noexcept;
};

}  // namespace newton_fractal

namespace nf = newton_fractal;

#endif  // NEWTON_FRACTAL_ZOOM_OBJECT_CREATOR_H
