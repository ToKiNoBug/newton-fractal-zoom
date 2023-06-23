//
// Created by David on 2023/6/21.
//

#include <memory>
#include <fractal_colors.h>
#include "gpu_interface.h"
#include <magic_enum.hpp>
#include <cuComplex.h>
// #include <cuda_wrappers/complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include "gpu_internal.h"

static_assert(cudaError_t::cudaSuccess == 0);

namespace newton_fractal {

constexpr int warp_size = 64;

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

 public:
  [[nodiscard]] tl::expected<void, std::string> run(
      const render_config_gpu_interface& config) & noexcept override;
};
}  // namespace newton_fractal

namespace nf = newton_fractal;
namespace nfi = newton_fractal::internal;

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

__global__ void norm_arg_cvt_and_minmax(double2* norm_and_arg,
                                        double* dest_norm_min,
                                        double* dest_norm_max,
                                        double* dest_arg_min,
                                        double* dest_arg_max) {
  const uint32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  double2& dst = norm_and_arg[global_idx];
  {
    const cuDoubleComplex src = reinterpret_cast<cuDoubleComplex&>(dst);
    dst.x = std::sqrt(src.x * src.x + src.y * src.y);
    dst.y = std::atan2(src.y, src.x);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    double norm_min{INFINITY}, norm_max{-INFINITY};
    double arg_min{INFINITY}, arg_max{-INFINITY};
    for (uint32_t offset = 0; offset < blockDim.x; offset++) {
      const double2 norm_arg = norm_and_arg[global_idx + offset];
      const double norm = norm_arg.x;
      const double arg = norm_arg.y;

      norm_min = std::min(norm, norm_min);
      norm_max = std::max(norm, norm_max);

      arg_min = std::min(arg, arg_min);
      arg_max = std::max(arg, arg_max);
    }
    dest_norm_min[blockIdx.x] = norm_min;
    dest_norm_max[blockIdx.x] = norm_max;

    dest_arg_min[blockIdx.x] = arg_min;
    dest_arg_max[blockIdx.x] = arg_max;
  }
}

struct normalize_option {
  double min;
  double max;

  NF_HOST_DEVICE_FUN inline double normalize(double src) const noexcept {
    assert(src >= this->min);
    assert(src <= this->max);
    assert(this->max != this->min);
    return (src - this->min) / (this->max - this->min);
  }
};

__global__ void run_normalization(double2* norm_and_arg, bool normalize_norm,
                                  normalize_option norm_option,
                                  bool normalize_arg,
                                  normalize_option arg_option) {
  const uint32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  double2& dst = norm_and_arg[global_idx];

  if (normalize_norm) {
    dst.x = norm_option.normalize(dst.x);
  }
  if (normalize_arg) {
    dst.y = arg_option.normalize(dst.y);
  }
}

__global__ void run_render_1d(
    const nf::render_config::render_method* method_ptr,
    fractal_utils::pixel_RGB color_for_nan, const bool* has_value,
    const int* nearest_idx, double2* norm_and_arg_normalized) {}

tl::expected<void, std::string> nf::gpu_implementation::run(
    const render_config_gpu_interface& config) & noexcept {
  if (!config.ok()) {
    return tl::make_unexpected("The passed config is not ok.");
  }
  return {};
}