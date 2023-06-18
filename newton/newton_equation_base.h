//
// Created by joseph on 6/18/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_BASE_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_BASE_H

#include <vector>
#include <span>
#include <any>
#include <string>
#include <tl/expected.hpp>

namespace newton_fractal {

class newton_equation_base {
 public:
  virtual ~newton_equation_base() = default;

  [[nodiscard]] virtual int order() const noexcept = 0;

  [[nodiscard]] virtual std::string to_string() const noexcept = 0;

  virtual void iterate_n(std::any& z, int n) const noexcept = 0;
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_BASE_H
