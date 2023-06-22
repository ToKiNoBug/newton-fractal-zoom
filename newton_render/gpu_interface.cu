//
// Created by David on 2023/6/21.
//

#include <memory>
#include <fractal_colors.h>
#include "gpu_interface.h"
#include <magic_enum.hpp>
// #include <cuda_wrappers/complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>

static_assert(cudaError_t::cudaSuccess == 0);

namespace newton_fractal {

constexpr int warp_size = 64;

namespace internal {

template <typename T>
struct cuda_deleter {
 public:
  void operator()(T* data) const noexcept;
};
template <>
struct cuda_deleter<void> {
  void operator()(void* data) const noexcept;
};
template <typename T>
void cuda_deleter<T>::operator()(T* data) const noexcept {
  cuda_deleter<void>()(data);
}

template <typename T>
using unique_cu_ptr = std::unique_ptr<T, cuda_deleter<T>>;

struct pinned_deleter {
  void operator()(void* ptr) const noexcept { cudaFreeHost(ptr); }
};

}  // namespace internal

class gpu_implementation : public gpu_interface {
 private:
  internal::unique_cu_ptr<bool> m_has_value{nullptr};
  internal::unique_cu_ptr<uint8_t> m_nearest_index{nullptr};
  internal::unique_cu_ptr<std::complex<double>> m_complex_difference{nullptr};
  internal::unique_cu_ptr<fractal_utils::pixel_RGB> m_pixel{nullptr};

  std::unique_ptr<void, internal::pinned_deleter> m_pinned_buffer{nullptr};
  cudaStream_t m_stream{nullptr};

  int m_rows{0};
  int m_cols{0};

  int m_cuda_error_code{0};

 public:
  // gpu_implementation() = default;
  gpu_implementation(gpu_implementation&&) noexcept = default;
  gpu_implementation(const gpu_implementation&) = delete;
  gpu_implementation(int _r, int _c);
  ~gpu_implementation() override;

  [[nodiscard]] inline int rows() const noexcept override {
    return this->m_rows;
  }
  [[nodiscard]] inline int cols() const noexcept override {
    return this->m_cols;
  }

  [[nodiscard]] inline bool ok() const noexcept override {
    return this->m_cuda_error_code == 0;
  }
  [[nodiscard]] inline int error_code() const noexcept override {
    return this->m_cuda_error_code;
  }

  [[nodiscard]] tl::expected<void, std::string> set_has_value(
      std::span<const bool> src) & noexcept override;
  [[nodiscard]] tl::expected<void, std::string> set_nearest_index(
      std::span<const uint8_t> src) & noexcept override;
  [[nodiscard]] tl::expected<void, std::string> set_complex_difference(
      std::span<const std::complex<double>> src) & noexcept override;

  [[nodiscard]] inline int cpu_task_count() const noexcept {
    return this->size() % warp_size;
  }

  [[nodiscard]] inline int gpu_task_count() const noexcept {
    return this->size() - this->cpu_task_count();
  }

  [[nodiscard]] inline size_t required_pinned_memory() const noexcept {
    return this->gpu_task_count() * sizeof(std::complex<double>);
  }

 protected:
  tl::expected<void, std::string> copy_to_device(void* dst, size_t dst_bytes,
                                                 const void* src,
                                                 size_t src_bytes) & noexcept;
};
}  // namespace newton_fractal

namespace nf = newton_fractal;
namespace nfi = newton_fractal::internal;

template <typename T>
tl::expected<nfi::unique_cu_ptr<T>, cudaError_t> allocate_device_memory(
    size_t num_elements) noexcept {
  T* temp{nullptr};
  auto err = cudaMalloc(&temp, num_elements * sizeof(T));
  if (err != cudaError_t::cudaSuccess) {
    return tl::make_unexpected(err);
  }
  nfi::unique_cu_ptr<T> ret{temp};
  return ret;
}

nf::gpu_implementation::gpu_implementation(int _r, int _c)
    : m_rows{_r}, m_cols{_c} {
  {
    this->m_cuda_error_code = cudaStreamCreate(&this->m_stream);
    if (!this->ok()) return;
  }

#define NF_PRIVATE_MACRO_ALLOCATE_MEM(type, member)         \
  {                                                         \
    auto temp = allocate_device_memory<type>(this->size()); \
    if (temp.has_value()) {                                 \
      this->member = std::move(temp.value());               \
    } else {                                                \
      this->m_cuda_error_code = temp.error();               \
      return;                                               \
    }                                                       \
  }

  NF_PRIVATE_MACRO_ALLOCATE_MEM(bool, m_has_value);
  NF_PRIVATE_MACRO_ALLOCATE_MEM(uint8_t, m_nearest_index);
  NF_PRIVATE_MACRO_ALLOCATE_MEM(std::complex<double>, m_complex_difference);
  NF_PRIVATE_MACRO_ALLOCATE_MEM(fractal_utils::pixel_RGB, m_pixel);

  {
    void* temp{nullptr};
    this->m_cuda_error_code = cudaHostAlloc(
        &temp, this->required_pinned_memory(), cudaHostAllocDefault);
    if (!this->ok()) return;
    this->m_pinned_buffer.reset(temp);
  }
}

tl::expected<std::unique_ptr<nf::gpu_interface>, std::string>
nf::gpu_interface::create(int rows, int cols) noexcept {
  auto ret = std::make_unique<nf::gpu_implementation>(rows, cols);
  if (!ret->ok()) {
    return tl::make_unexpected(
        std::string{"Constructor failed with cuda error code "} +
        std::to_string(ret->error_code()));
  }
  return ret;
}

nf::gpu_implementation::~gpu_implementation() {
  cudaStreamDestroy(this->m_stream);
}

void nfi::cuda_deleter<void>::operator()(void* data) const noexcept {
  cudaFree(data);
}

tl::expected<void, std::string> nf::gpu_implementation::copy_to_device(
    void* dst, size_t dst_bytes, const void* src, size_t src_bytes) & noexcept {
  if (src_bytes != dst_bytes) {
    return tl::make_unexpected("The size of source and dest mismatch.");
  }
  if (src_bytes > this->required_pinned_memory()) {
    return tl::make_unexpected("Not enough pinned memory buffer.");
  }

  memcpy(this->m_pinned_buffer.get(), src, src_bytes);

  auto err =
      cudaMemcpyAsync(dst, this->m_pinned_buffer.get(), src_bytes,
                      cudaMemcpyKind::cudaMemcpyHostToDevice, this->m_stream);
  if (err != cudaError_t::cudaSuccess) {
    return tl::make_unexpected(
        std::string{"Failed to copy host data to device with error code "} +
        std::to_string(err));
  }
  return {};
}

tl::expected<void, std::string> nf::gpu_implementation::set_has_value(
    std::span<const bool> src) & noexcept {
  return this->copy_to_device(this->m_has_value.get(),
                              this->size() * sizeof(bool), src.data(),
                              src.size_bytes());
}

[[nodiscard]] tl::expected<void, std::string>
nf::gpu_implementation::set_nearest_index(
    std::span<const uint8_t> src) & noexcept {
  return this->copy_to_device(this->m_nearest_index.get(),
                              this->size() * sizeof(uint8_t), src.data(),
                              src.size_bytes());
}

[[nodiscard]] tl::expected<void, std::string>
nf::gpu_implementation::set_complex_difference(
    std::span<const std::complex<double>> src) & noexcept {
  return this->copy_to_device(this->m_complex_difference.get(),
                              this->size() * sizeof(std::complex<double>),
                              src.data(), src.size_bytes());
}
