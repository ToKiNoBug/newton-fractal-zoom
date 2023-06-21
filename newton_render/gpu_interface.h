//
// Created by David on 2023/6/21.
//

#ifndef NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H
#define NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H

#include <memory>
#include <complex>
#include <tl/expected.hpp>
#include <string>
#include <optional>
#include <fractal_colors.h>
#include <span>

namespace newton_fractal {
namespace internal {

template <typename T>
class cuda_deleter {
 public:
  void operator()(T* data) const noexcept;
};
template <>
class cuda_deleter<void> {
 public:
  // implemented in gpu_interface.cu
  void operator()(void* data) const noexcept;
};
template <typename T>
void cuda_deleter<T>::operator()(T* data) const noexcept {
  cuda_deleter<void>()(data);
}

template <typename T>
using unique_cu_ptr = std::unique_ptr<T, cuda_deleter<T>>;

}  // namespace internal
class gpu_interface {
 private:
  internal::unique_cu_ptr<bool> m_has_value{nullptr};
  internal::unique_cu_ptr<uint8_t> m_nearest_index{nullptr};
  internal::unique_cu_ptr<std::complex<double>> m_complex_difference{nullptr};
  internal::unique_cu_ptr<fractal_utils::pixel_RGB> m_pixel{nullptr};

  int m_rows{0};
  int m_cols{0};

  int m_cuda_error_code{0};

 public:
  gpu_interface() = default;
  gpu_interface(gpu_interface&&) noexcept = default;
  gpu_interface(const gpu_interface&) = delete;
  ~gpu_interface() = default;

  gpu_interface(int _r, int _c);

  [[nodiscard]] inline int rows() const noexcept { return this->m_rows; }
  [[nodiscard]] inline int cols() const noexcept { return this->m_cols; }
  [[nodiscard]] inline int size() const noexcept {
    return this->m_rows * this->m_cols;
  }

  [[nodiscard]] inline bool ok() const noexcept {
    return this->m_cuda_error_code == 0;
  }
  [[nodiscard]] inline auto error_code() const noexcept {
    return this->m_cuda_error_code;
  }

  [[nodiscard]] tl::expected<void, std::string> set_has_value(
      std::span<const bool> src) & noexcept;
  [[nodiscard]] tl::expected<void, std::string> set_nearest_index(
      std::span<const uint8_t> src) & noexcept;
  [[nodiscard]] tl::expected<void, std::string> set_complex_difference(
      std::span<const std::complex<double>> src) & noexcept;
};
}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_GPU_INTERFACE_H
