//
// Created by David on 2023/6/21.
//

#include "gpu_interface.h"
#include <memory>
#include <fractal_colors.h>
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
  bool m_require_norm_arg_cvt{true};

 public:
  // gpu_implementation() = default;
  // gpu_implementation(gpu_implementation&&) noexcept = default;
  // gpu_implementation(const gpu_implementation&) = delete;
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

  [[nodiscard]] inline int gpu_thread_count() const noexcept {
    if (this->size() % warp_size == 0) {
      return this->size();
    }
    int ret = this->size() / warp_size;
    ret = (ret + 1) * warp_size;
    return ret;
  }

  [[nodiscard]] inline int gpu_block_count() const noexcept {
    assert(this->gpu_thread_count() % 32 == 0);
    return this->gpu_thread_count() / 32;
  }

  [[nodiscard]] inline size_t required_pinned_memory() const noexcept {
    return this->size() * (sizeof(std::complex<double>) + sizeof(bool));
  }

 protected:
  tl::expected<void, std::string> copy_to_device(void* dst, size_t dst_bytes,
                                                 const void* src,
                                                 size_t src_bytes) & noexcept;

  tl::expected<void, std::string> compute_minmax(int skip_rows, int skip_cols,
                                                 double& mag_min,
                                                 double& mag_max,
                                                 double& arg_min,
                                                 double& arg_max) & noexcept;

 public:
  [[nodiscard]] tl::expected<void, std::string> run(
      const render_config_gpu_interface& config, int skip_rows, int skip_cols,
      bool sync) & noexcept final;

  [[nodiscard]] tl::expected<void, std::string> get_pixels(
      fractal_utils::map_view dest) & noexcept final;
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
  this->m_require_norm_arg_cvt = true;
  return this->copy_to_device(this->m_complex_difference.get(),
                              this->size() * sizeof(std::complex<double>),
                              src.data(), src.size_bytes());
}

struct coordinate_t {
  int row;
  int col;
};
__host__ __device__ coordinate_t compute_idx(int rows, int cols, int skip_rows,
                                             int skip_cols,
                                             int global_idx) noexcept {
  const int actual_rows = rows - 2 * skip_rows;
  const int actual_cols = cols - 2 * skip_cols;
  assert(actual_rows > 0);
  assert(actual_cols > 0);
  assert(global_idx >= 0);
  assert(global_idx < actual_rows * actual_cols);

  const int r_offset = global_idx / actual_cols;
  const int c_offset = global_idx % actual_cols;
  return coordinate_t{r_offset + skip_rows, c_offset + skip_cols};
}

__host__ __device__ int index_1d_of_coordinate(int rows, int cols,
                                               coordinate_t cd) noexcept {
  assert(cd.row >= 0);
  assert(cd.col >= 0);
  assert(cd.row < rows);
  assert(cd.col < cols);
  return cd.row * cols + cd.col;
}

__global__ void norm_arg_cvt(double2* norm_and_arg, int rows, int cols) {
  const uint32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (global_idx >= rows * cols) {
    return;
  }

  double2& dst = norm_and_arg[global_idx];
  {
    const cuDoubleComplex src = reinterpret_cast<cuDoubleComplex&>(dst);
    dst.x = std::sqrt(src.x * src.x + src.y * src.y);
    dst.y = std::atan2(src.y, src.x);
  }
}

struct normalize_option {
  double min;
  double max;

  __host__ __device__ inline double normalize(double src) const noexcept {
    assert(src >= this->min);
    assert(src <= this->max);
    assert(this->max != this->min);
    return (src - this->min) / (this->max - this->min);
  }
};

__global__ void run_render_1d(
    const nf::render_config::render_method* method_ptr,
    fractal_utils::pixel_RGB color_for_nan, const bool* has_value,
    const uint8_t* nearest_idx, const double2* norm_and_arg,
    fractal_utils::pixel_RGB* dst_u8c3, coordinate_t size, coordinate_t skip,
    normalize_option norm_option, normalize_option arg_option) {
  const uint32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_idx >=
      (size.row - 2 * skip.row) * (size.col - 2 * skip.col)) {
    return;
  }

  const auto coordinate =
      compute_idx(size.row, size.col, skip.row, skip.col, global_thread_idx);
  const int global_offset =
      index_1d_of_coordinate(size.row, size.col, coordinate);

  if (!has_value[global_offset]) {
    dst_u8c3[global_offset] = color_for_nan;
    return;
  }

  const float normalized_norm =
      (float)norm_option.normalize(norm_and_arg[global_offset].x);
  const float normalized_arg =
      (float)arg_option.normalize(norm_and_arg[global_offset].y);

  auto color =
      nf::render(method_ptr, color_for_nan, has_value[global_offset],
                 nearest_idx[global_offset], normalized_norm, normalized_arg);

  dst_u8c3[global_offset] = color;
}

tl::expected<void, std::string> nf::gpu_implementation::compute_minmax(
    int skip_rows, int skip_cols, double& mag_min, double& mag_max,
    double& arg_min, double& arg_max) & noexcept {
  cudaError_t err{cudaSuccess};
  // compute min and max

  uint8_t* const hostptr_norm_arg = (uint8_t*)this->m_pinned_buffer.get();
  uint8_t* const hostptr_has_value =
      hostptr_norm_arg + this->size() * sizeof(std::complex<double>);
  assert(hostptr_has_value + this->size() * sizeof(bool) <=
         (uint8_t*)this->m_pinned_buffer.get() +
             this->required_pinned_memory());
  err = cudaMemcpyAsync(hostptr_norm_arg, this->m_complex_difference.get(),
                        this->size() * sizeof(std::complex<double>),
                        cudaMemcpyKind::cudaMemcpyDeviceToHost, this->m_stream);
  if (err != cudaSuccess) {
    return tl::make_unexpected(
        std::string{"cudaMemcpyAsync failed with error code "} +
        std::to_string(err));
  }
  err = cudaMemcpyAsync(hostptr_has_value, this->m_has_value.get(),
                        this->size() * sizeof(bool),
                        cudaMemcpyKind::cudaMemcpyDeviceToHost, this->m_stream);
  if (err != cudaSuccess) {
    return tl::make_unexpected(
        std::string{"cudaMemcpyAsync failed with error code "} +
        std::to_string(err));
  }
  err = cudaStreamSynchronize(this->m_stream);
  if (err != cudaSuccess) {
    return tl::make_unexpected(
        std::string{"cudaStreamSynchronize failed with error code "} +
        std::to_string(err));
  }

  mag_min = INFINITY;
  arg_min = INFINITY;
  mag_max = -INFINITY;
  mag_max = -INFINITY;

  const bool* const has_value_ptr = reinterpret_cast<bool*>(hostptr_has_value);
  const double2* const mag_arg_ptr =
      reinterpret_cast<double2*>(hostptr_norm_arg);

  int computed_pixels{0};
  for (int r = skip_rows; r < this->rows() - skip_rows; r++) {
    for (int c = skip_cols; c < this->cols() - skip_cols; c++) {
      const int offset =
          index_1d_of_coordinate(this->rows(), this->cols(), {r, c});
      if (!has_value_ptr[offset]) [[unlikely]] {
        continue;
      }
      computed_pixels++;

      const double mag = mag_arg_ptr[offset].x;
      const double arg = mag_arg_ptr[offset].y;
      mag_max = std::max(mag, mag_max);
      mag_min = std::min(mag, mag_min);
      arg_max = std::max(arg, arg_max);
      arg_min = std::min(arg, arg_min);
    }
  }

  if (computed_pixels <= 0) [[unlikely]] {
    mag_min = 0;
    arg_min = 0;
    mag_max = 1;
    arg_max = 1;
    return {};
  }

  if (computed_pixels <= 1) [[unlikely]] {
    mag_max = mag_min + 1;
    arg_max = arg_min + 1;
    return {};
  }
  return {};
}

tl::expected<void, std::string> nf::gpu_implementation::run(
    const render_config_gpu_interface& config, int skip_rows, int skip_cols,
    bool sync) & noexcept {
  if (!config.ok()) {
    return tl::make_unexpected("The passed config is not ok.");
  }

  cudaError_t err{cudaSuccess};

  if (this->m_require_norm_arg_cvt) {
    norm_arg_cvt<<<this->gpu_block_count(), warp_size, 0, this->m_stream>>>(
        (double2*)this->m_complex_difference.get(), this->rows(), this->cols());
    this->m_require_norm_arg_cvt = false;
  }

  normalize_option mag{}, arg{};
  {
    auto temp = this->compute_minmax(skip_rows, skip_cols, mag.min, mag.max,
                                     arg.min, arg.max);
    if (!temp.has_value()) {
      return tl::make_unexpected(
          std::string{"Failed to compute min and max value. Detail: "} +
          temp.error());
    }
  }

  {
    run_render_1d<<<this->gpu_block_count(), warp_size, 0, this->m_stream>>>(
        config.method_ptr(), config.color_for_nan(), this->m_has_value.get(),
        this->m_nearest_index.get(),
        (const double2*)this->m_complex_difference.get(), this->m_pixel.get(),
        {this->rows(), this->cols()}, {skip_rows, skip_cols}, mag, arg);
  }

  if (sync) {
    err = cudaStreamSynchronize(this->m_stream);
    if (err != cudaSuccess) {
      return tl::make_unexpected(
          std::string{"cudaStreamSynchronize failed with error code "} +
          std::to_string(err));
    }
  }

  return {};
}

tl::expected<void, std::string> nf::gpu_implementation::get_pixels(
    fractal_utils::map_view img_u8c3) & noexcept {
  if (img_u8c3.element_bytes() != 3) {
    return tl::make_unexpected("The element bytes of dest is not 3.");
  }

  if (img_u8c3.rows() != this->rows() || img_u8c3.cols() != this->cols()) {
    return tl::make_unexpected("The image size mismatch.");
  }

  auto err =
      cudaMemcpyAsync(this->m_pinned_buffer.get(), this->m_pixel.get(),
                      this->size() * sizeof(fractal_utils::pixel_RGB),
                      cudaMemcpyKind::cudaMemcpyDeviceToHost, this->m_stream);
  if (err != cudaSuccess) {
    return tl::make_unexpected(
        std::string{"cudaMemcpyAsync failed with error code "} +
        std::to_string(err));
  }

  err = cudaStreamSynchronize(this->m_stream);
  if (err != cudaSuccess) {
    return tl::make_unexpected(
        std::string{"cudaStreamSynchronize failed with error code "} +
        std::to_string(err));
  }
  memcpy(img_u8c3.data(), this->m_pinned_buffer.get(), img_u8c3.bytes());
  return {};
}