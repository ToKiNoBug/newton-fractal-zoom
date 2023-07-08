#include "gpu_interface.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_new_allocator.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <exception>
#include <thrust/universal_allocator.h>
#include <limits>
#include <variant>
#include "cpu_renderer.h"

namespace newton_fractal {

constexpr int ceil_up_to(int x, int divide) {
  if (x % divide == 0) {
    return x;
  }
  const int a = x / divide;
  return (a + 1) * divide;
}

struct cuda_stream_deleter {
  void operator()(cudaStream_t s) const { cudaStreamDestroy(s); }
};

std::runtime_error create_cuda_error(const std::string& fun,
                                     cudaError_t err) noexcept {
  return std::runtime_error{
      fun + " failed with cuda error code " + std::to_string(err) + " aka " +
      cudaGetErrorName(err) + ", description: " + cudaGetErrorString(err)};
}

void handel_error(const std::string& fun, cudaError_t err) noexcept(false) {
  if (err != cudaSuccess) {
    throw create_cuda_error(fun, err);
  }
}

struct coordinate_t {
  int row;
  int col;
};

__host__ __device__ coordinate_t coordinate_of_index(int rows, int cols,
                                                     int skip_rows,
                                                     int skip_cols,
                                                     int global_idx) noexcept {
  [[maybe_unused]] const int actual_rows = rows - 2 * skip_rows;
  const int actual_cols = cols - 2 * skip_cols;
  assert(actual_rows > 0);
  assert(actual_cols > 0);
  assert(global_idx >= 0);
  assert(global_idx < actual_rows * actual_cols);

  const int r_offset = global_idx / actual_cols;
  const int c_offset = global_idx % actual_cols;
  return coordinate_t{r_offset + skip_rows, c_offset + skip_cols};
}

__host__ __device__ int global_index_of_coordinate(int rows, int cols,
                                                   coordinate_t cd) noexcept {
  assert(cd.row >= 0);
  assert(cd.col >= 0);
  assert(cd.row < rows);
  assert(cd.col < cols);
  return cd.row * cols + cd.col;
}

static_assert(sizeof(thrust::complex<double>) == sizeof(std::complex<double>));

struct normalizer {
  double min{std::numeric_limits<double>::infinity()};
  double max{-std::numeric_limits<double>::infinity()};

  void add_data(double d) & noexcept {
    this->min = std::min(d, this->min);
    this->max = std::max(d, this->max);
  }

  void reset() & noexcept {
    this->min = std::numeric_limits<double>::infinity();
    this->max = -std::numeric_limits<double>::infinity();
  }

  NF_HOST_DEVICE_FUN double operator()(double src) const noexcept {
    return (src - this->min) / (this->max - this->min);
  }
};

class gpu_render_impl : public gpu_render {
 private:
  thrust::device_vector<bool> m_has_value;
  thrust::device_vector<uint8_t> m_nearest_index;
  thrust::device_vector<thrust::complex<double>> m_mag_arg;
  thrust::device_vector<fu::pixel_RGB> m_image;

  thrust::host_vector<uint8_t, thrust::universal_allocator<uint8_t>>
      m_pinned_buffer;
  thrust::host_vector<uint8_t, thrust::universal_allocator<uint8_t>>
      m_pinned_buffer_image;

  std::unique_ptr<CUstream_st, cuda_stream_deleter> m_stream{nullptr};

  std::optional<
      std::variant<fractal_utils::unique_map, fractal_utils::constant_view>>
      m_has_value_host{std::nullopt};
  std::optional<uint8_t> m_nearest_index_max{std::nullopt};
  // std::optional<std::pair<int, int>> m_size_rc{std::nullopt};

  template <typename T>
  void copy_to_device(
      fu::constant_view src,
      thrust::host_vector<uint8_t, thrust::universal_allocator<uint8_t>>&
          pinned_buffer,
      thrust::device_vector<T>& dest) & noexcept(false) {
    if (src.element_bytes() != sizeof(T)) {
      throw std::runtime_error{"Element size of matrix and dest mismatch."};
    }
    pinned_buffer.resize(src.bytes());
    memcpy(pinned_buffer.data().get(), src.data(), src.bytes());
    dest.resize(src.size());
    auto err = cudaMemcpyAsync(
        dest.data().get(), pinned_buffer.data().get(), src.bytes(),
        cudaMemcpyKind::cudaMemcpyHostToDevice, this->m_stream.get());
    handel_error("cudaMemcpyAsync", err);
    this->wait_for_finished();
  }

  template <typename T>
  void copy_to_device(fu::constant_view src,
                      thrust::device_vector<T>& dest) & noexcept(false) {
    this->copy_to_device(src, this->m_pinned_buffer, dest);
  }

  template <typename T>
  void load_to_pinned_buffer(
      const thrust::device_vector<T>& src,
      thrust::host_vector<uint8_t, thrust::universal_allocator<uint8_t>>&
          pinned_buffer) & noexcept(false) {
    pinned_buffer.resize(src.size() * sizeof(T));
    auto err = cudaMemcpyAsync(pinned_buffer.data().get(), src.data().get(),
                               src.size() * sizeof(T), cudaMemcpyDeviceToHost,
                               this->m_stream.get());
    handel_error("cudaMemcpyAsync", err);
    this->wait_for_finished();
  }

  template <typename T>
  void copy_to_host(
      const thrust::device_vector<T>& src,
      thrust::host_vector<uint8_t, thrust::universal_allocator<uint8_t>>&
          pinned_buffer,
      std::span<T> dest) & noexcept(false) {
    if (src.size() != dest.size()) {
      throw std::runtime_error{"The size of source and dest mismatch."};
    }
    this->load_to_pinned_buffer(src, pinned_buffer);
    memcpy(dest.data(), pinned_buffer.data().get(), dest.size_bytes());
  }

  template <typename T>
  void copy_to_host(const thrust::device_vector<T>& src,
                    std::span<T> dest) & noexcept(false) {
    this->copy_to_host(src, this->m_pinned_buffer, dest);
  }

  void wait_for_finished() & noexcept(false);

 public:
  explicit gpu_render_impl(int size);

  [[nodiscard]] tl::expected<void, std::string> set_data(
      fractal_utils::constant_view has_value,
      fractal_utils::constant_view nearest_index,
      fractal_utils::constant_view map_complex_difference,
      bool deep_copy) & noexcept final;

  [[nodiscard]] tl::expected<fractal_utils::constant_view, std::string> render(
      const render_config_gpu_interface& config, int skip_rows,
      int skip_cols) & noexcept final;

  [[nodiscard]] tl::expected<void, std::string> render(
      const render_config_gpu_interface& config, fu::constant_view has_value,
      fu::constant_view nearest_index, fu::constant_view complex_difference,
      fu::map_view image_u8c3, int skip_rows, int skip_cols) & noexcept final;
};

gpu_render_impl::gpu_render_impl(int size) {
  {
    cudaStream_t temp;
    auto err = cudaStreamCreate(&temp);
    handel_error("cudaStreamCreate", err);
    this->m_stream.reset(temp);
  }
  this->m_has_value.reserve(size);
  this->m_nearest_index.reserve(size);
  this->m_mag_arg.reserve(size);
  this->m_image.reserve(size);

  this->m_pinned_buffer.reserve(size * sizeof(thrust::complex<double>));
  this->m_pinned_buffer_image.reserve(size * sizeof(fu::pixel_RGB));
}

tl::expected<std::unique_ptr<gpu_render>, std::string> gpu_render::create(
    int rows, int cols) noexcept {
  std::unique_ptr<gpu_render> ret{nullptr};
  try {
    ret = std::make_unique<gpu_render_impl>(rows * cols);
  } catch (std::exception& e) {
    return tl::make_unexpected(e.what());
  }
  return ret;
}

void gpu_render_impl::wait_for_finished() & noexcept(false) {
  auto err = cudaStreamSynchronize(this->m_stream.get());
  handel_error("cudaStreamSynchronize", err);
}

__global__ void complex_norm_arg_cvt(thrust::complex<double>* mag_arg,
                                     const bool* has_value, int rows, int cols);

void find_min_max(fu::constant_view has_value, fu::constant_view mag_arg,
                  int skip_rows, int skip_cols, normalizer& mag,
                  normalizer& arg) noexcept;

__global__ void render_image(
    const bool* has_value, const uint8_t* nearest_index,
    const thrust::complex<double>*, fu::pixel_RGB* image_u8c3,
    const normalizer mag_normalizer, const normalizer arg_normalizer,
    fu::pixel_RGB color_nan, const render_config::render_method* methods,
    int rows, int cols, int skip_rows, int skip_cols);

tl::expected<void, std::string> gpu_render_impl::set_data(
    fractal_utils::constant_view has_value,
    fractal_utils::constant_view nearest_index,
    fractal_utils::constant_view complex_difference,
    bool deep_copy) & noexcept {
  if (has_value.rows() != nearest_index.rows() ||
      has_value.cols() != nearest_index.cols()) {
    return tl::make_unexpected(
        "The matrix size of has_value and nearest_index mismatch.");
  }
  if (nearest_index.rows() != complex_difference.rows() ||
      nearest_index.cols() != complex_difference.cols()) {
    return tl::make_unexpected(
        "The matrix size of nearest_index and complex_difference mismatch.");
  }

  const int rows = has_value.rows();
  const int cols = has_value.cols();
  const int global_size = has_value.size();

  constexpr int warp_size = 32;

  const int global_required_threads = ceil_up_to(global_size, warp_size);
  const int global_required_blocks = global_required_threads / warp_size;
  assert(global_required_threads % warp_size == 0);

  try {
    // this->m_size_rc = {rows, cols};
    this->m_nearest_index_max =
        internal::compute_max_nearest_index(has_value, nearest_index);

    internal::set_data(has_value, this->m_has_value_host, deep_copy);

    this->copy_to_device(has_value, this->m_has_value);
    this->copy_to_device(nearest_index, this->m_nearest_index);
    this->copy_to_device(complex_difference, this->m_mag_arg);

    complex_norm_arg_cvt<<<global_required_blocks, warp_size, 0,
                           this->m_stream.get()>>>(
        this->m_mag_arg.data().get(), this->m_has_value.data().get(), rows,
        cols);
    this->load_to_pinned_buffer(this->m_mag_arg, this->m_pinned_buffer);
  } catch (std::exception& e) {
    return tl::make_unexpected(e.what());
  }
  return {};

  //  if (image_u8c3.rows() != complex_difference.rows() ||
  //      image_u8c3.cols() != complex_difference.cols()) {
  //    return tl::make_unexpected(
  //        "The matrix size of image_u8c3 and complex_difference mismatch.");
  //  }
}

[[nodiscard]] tl::expected<fractal_utils::constant_view, std::string>
gpu_render_impl::render(const render_config_gpu_interface& config,
                        int skip_rows, int skip_cols) & noexcept {
  try {
    const size_t rows = internal::get_map(this->m_has_value_host).rows();
    const size_t cols = internal::get_map(this->m_has_value_host).cols();
    const size_t global_size = internal::get_map(this->m_has_value_host).size();

    constexpr size_t warp_size = 32;
    const size_t size = (rows - 2 * skip_rows) * (cols - 2 * skip_cols);
    const size_t required_threads = ceil_up_to(size, warp_size);
    const size_t required_blocks = required_threads / warp_size;
    assert(required_threads % warp_size == 0);

    // compute the range
    normalizer mag, arg;
    {
      fu::constant_view cv_mag_arg{
          (const void*)this->m_pinned_buffer.data().get(), (size_t)rows,
          (size_t)cols, sizeof(std::complex<double>)};
      find_min_max(internal::get_map(this->m_has_value_host), cv_mag_arg,
                   skip_rows, skip_cols, mag, arg);
    }

    this->m_image.resize(global_size);
    render_image<<<required_blocks, warp_size, 0, this->m_stream.get()>>>(
        this->m_has_value.data().get(), this->m_nearest_index.data().get(),
        this->m_mag_arg.data().get(), this->m_image.data().get(), mag, arg,
        config.color_for_nan(), config.method_ptr(), rows, cols, skip_rows,
        skip_cols);
    this->load_to_pinned_buffer(this->m_image, this->m_pinned_buffer_image);

    return fu::constant_view{
        (const void*)this->m_pinned_buffer_image.data().get(), rows, cols,
        sizeof(fu::pixel_RGB)};
  } catch (std::exception& e) {
    return tl::make_unexpected(e.what());
  }
}

tl::expected<void, std::string> gpu_render_impl::render(
    const render_config_gpu_interface& config, fu::constant_view has_value,
    fu::constant_view nearest_index, fu::constant_view complex_difference,
    fu::map_view image_u8c3, int skip_rows, int skip_cols) & noexcept {
  auto err =
      this->set_data(has_value, nearest_index, complex_difference, false);
  if (!err) {
    return tl::make_unexpected(err.error());
  }
  err = gpu_render::render(config, image_u8c3, skip_rows, skip_cols);
  if (!err) {
    return tl::make_unexpected(err.error());
  }
  return {};
}

tl::expected<void, std::string> gpu_render::render(
    const render_config_gpu_interface& config,
    fractal_utils::map_view image_u8c3, int skip_rows,
    int skip_cols) & noexcept {
  auto result = this->render(config, skip_rows, skip_cols);
  if (!result) {
    return tl::make_unexpected(result.error());
  }

  if (image_u8c3 != result.value()) {
    return tl::make_unexpected(
        "The size of image_u8c3 mismatch with other given matrices.");
  }
  memcpy(image_u8c3.data(), result.value().data(), result.value().bytes());
  return {};
}

__global__ void complex_norm_arg_cvt(thrust::complex<double>* mag_arg,
                                     const bool* has_value, int rows,
                                     int cols) {
  const uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_index >= rows * cols) {
    return;
  }
  if (!has_value[global_index]) {
    return;
  }
  const thrust::complex<double> src = mag_arg[global_index];
  const double mag = abs(src);
  const double angle = arg(src);

  mag_arg[global_index] = thrust::complex<double>{mag, angle};
}

void find_min_max(fu::constant_view has_value, fu::constant_view mag_arg,
                  int skip_rows, int skip_cols, normalizer& mag,
                  normalizer& arg) noexcept {
  assert(has_value.rows() == mag_arg.rows());
  assert(has_value.cols() == mag_arg.cols());
  mag.reset();
  arg.reset();

  int64_t counter{0};
  for (int r = skip_rows; r < mag_arg.rows() - skip_rows; r++) {
    for (int c = skip_cols; c < mag_arg.cols() - skip_cols; c++) {
      if (!has_value.at<bool>(r, c)) {
        continue;
      }
      const auto cplx = mag_arg.at<std::complex<double>>(r, c);
      mag.add_data(cplx.real());
      arg.add_data(cplx.imag());
      counter++;
    }
  }

  if (counter <= 0) {
    mag.min = 0;
    mag.max = 1;
    arg = mag;
    return;
  }
  if (counter == 1) {
    mag.max = mag.min + 1;
    arg.max = arg.min + 1;
  }
}

__global__ void render_image(
    const bool* has_value, const uint8_t* nearest_index,
    const thrust::complex<double>* mag_and_arg, fu::pixel_RGB* image_u8c3,
    const normalizer mag_normalizer, const normalizer arg_normalizer,
    fu::pixel_RGB color_nan, const render_config::render_method* methods,
    int rows, int cols, int skip_rows, int skip_cols) {
  const uint32_t thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_index >= (rows - 2 * skip_rows) * (cols - 2 * skip_cols)) {
    return;
  }

  const auto global_coord =
      coordinate_of_index(rows, cols, skip_rows, skip_cols, (int)thread_index);

  const auto mem_access_offset =
      global_index_of_coordinate(rows, cols, global_coord);

  const auto mag_arg = mag_and_arg[mem_access_offset];
  const double normalized_mag = mag_normalizer(mag_arg.real());
  const double normalized_arg = arg_normalizer(mag_arg.imag());

  image_u8c3[mem_access_offset] =
      render(methods, color_nan, has_value[mem_access_offset],
             nearest_index[mem_access_offset], (float)normalized_mag,
             (float)normalized_arg);
}
}  // namespace newton_fractal