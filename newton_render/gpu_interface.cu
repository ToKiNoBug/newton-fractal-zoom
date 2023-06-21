//
// Created by David on 2023/6/21.
//

#include "gpu_interface.h"
// #include <cuda_wrappers/complex>
#include <cuda.h>
#include <cuda_runtime.h>

static_assert(cudaError_t::cudaSuccess == 0);

namespace nf = newton_fractal;
namespace nfi = newton_fractal::internal;

void nfi::cuda_deleter<void>::operator()(void* data) const noexcept {
  cudaFree(data);
}

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

nf::gpu_interface::gpu_interface(int _r, int _c) : m_rows{_r}, m_cols{_c} {
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
}

tl::expected<void, std::string> copy_to_device(void* dst, size_t dst_bytes,
                                               const void* src,
                                               size_t src_bytes) noexcept {
  if (src_bytes != dst_bytes) {
    return tl::make_unexpected("The size of source and dest mismatch.");
  }
  auto err =
      cudaMemcpy(dst, src, src_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
  if (err != cudaError_t::cudaSuccess) {
    return tl::make_unexpected(
        std::string{"Failed to copy host data to device with error code "} +
        std::to_string(err));
  }
  return {};
}

tl::expected<void, std::string> nf::gpu_interface::set_has_value(
    std::span<const bool> src) & noexcept {
  return copy_to_device(this->m_has_value.get(), this->size() * sizeof(bool),
                        src.data(), src.size_bytes());
}

[[nodiscard]] tl::expected<void, std::string>
nf::gpu_interface::set_nearest_index(std::span<const uint8_t> src) & noexcept {
  return copy_to_device(this->m_nearest_index.get(),
                        this->size() * sizeof(uint8_t), src.data(),
                        src.size_bytes());
}

[[nodiscard]] tl::expected<void, std::string>
nf::gpu_interface::set_complex_difference(
    std::span<const std::complex<double>> src) & noexcept {
  return copy_to_device(this->m_complex_difference.get(),
                        this->size() * sizeof(std::complex<double>), src.data(),
                        src.size_bytes());
}
