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
#include <complex>
#include <optional>
#include <core_utils.h>
#include <memory>
#include <nlohmann/json.hpp>

using njson = nlohmann::json;

namespace newton_fractal {

enum class float_save_format : uint8_t {
  directly,
  hex_string,
  formatted_string
};

class newton_equation_base {
 public:
  virtual ~newton_equation_base() = default;

  [[nodiscard]] virtual int order() const noexcept = 0;

  [[nodiscard]] virtual std::string to_string() const noexcept = 0;

  [[nodiscard]] virtual std::complex<double> point_at(
      int idx) const noexcept = 0;

  virtual void reset(
      std::span<const std::complex<double>> points) & noexcept = 0;

  /*
  virtual void iterate_n(std::any &z, int iteration_times) const noexcept = 0;
  */

  struct single_result {
    int nearest_point_idx;
    std::complex<double> difference;
  };

  /*
  [[nodiscard]] virtual std::optional<single_result> compute_single(
      std::any &z, int iteration_times) const noexcept = 0;
  */

  struct compute_option {
    fractal_utils::map_view bool_has_result;
    fractal_utils::map_view u8_nearest_point_idx;
    fractal_utils::map_view f64complex_difference;
  };

  virtual void compute(const fractal_utils::wind_base &wind,
                       int iteration_times,
                       compute_option &opt) const noexcept = 0;

  virtual void clear() & noexcept = 0;

  [[nodiscard]] virtual njson::array_t to_json(
      float_save_format fsf) const noexcept = 0;

  [[nodiscard]] virtual std::unique_ptr<newton_equation_base> copy()
      const noexcept = 0;

 public:
  /* [[nodiscard]] virtual
   tl::expected<std::unique_ptr<fractal_utils::wind_base>, std::string>
   load_wind(const njson &) const noexcept = 0;

   */
};

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_EQUATION_BASE_H
