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
  enum class data_tag : int64_t {
    metadata = 0,
    map_has_result = 1,
    map_nearest_point_idx = 2,
    map_complex_difference = 3
  };

  newton_archive() = default;
  newton_archive(newton_archive &&) = default;
  newton_archive(const newton_archive &) = default;
  explicit newton_archive(const meta_data &) noexcept;
  explicit newton_archive(meta_data &&) noexcept;

  newton_archive &operator=(newton_archive &&) & noexcept = default;

  void setup_matrix() & noexcept;

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

  [[nodiscard]] tl::expected<void, std::string> save(
      std::ostream &os) const noexcept;

  [[nodiscard]] tl::expected<void, std::string> save(
      std::string_view filename) const noexcept;

  [[nodiscard]] tl::expected<void, std::string> save_raw(
      std::string_view filename) const noexcept;

  [[nodiscard]] tl::expected<void, std::string> load(
      std::istream &is, bool ignore_compute_objects,
      std::span<uint8_t> buffer) & noexcept;

  [[nodiscard]] tl::expected<void, std::string> load(
      std::string_view filename, bool ignore_compute_objects,
      std::span<uint8_t> buffer) & noexcept;

  [[nodiscard]] static tl::expected<newton_archive, std::string> load_archive(
      std::string_view filename, bool ignore_compute_objects,
      std::span<uint8_t> buffer) noexcept;
};

[[nodiscard]] size_t expected_buffer_size(int rows, int cols) noexcept;

tl::expected<void, std::string> check_archive(
    const newton_archive &ar) noexcept;
}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_ARCHIVE_H
