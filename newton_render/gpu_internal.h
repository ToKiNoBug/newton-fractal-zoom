//
// Created by David on 2023/6/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_GPU_INTERNAL_H
#define NEWTON_FRACTAL_ZOOM_GPU_INTERNAL_H

#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>

namespace newton_fractal {

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

template <typename T>
tl::expected<internal::unique_cu_ptr<T>, cudaError_t> allocate_device_memory(
    size_t num_elements) noexcept {
  T* temp{nullptr};
  auto err = cudaMalloc(&temp, num_elements * sizeof(T));
  if (err != cudaError_t::cudaSuccess) {
    return tl::make_unexpected(err);
  }
  internal::unique_cu_ptr<T> ret{temp};
  return ret;
}
}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_GPU_INTERNAL_H
