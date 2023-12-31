//
// Created by David on 2023/7/22.
//
#define JSON_HAS_RANGES false

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <exception>
#include <cuda.h>
#include <mutex>
#include "computation.hpp"
#include "cuda_support.cuh"
// #include "newton_equation.hpp"

namespace newton_fractal {

namespace internal {

void handle_error(const char *fun_name, cudaError_t error_code) {
  if (error_code != cudaSuccess) {
    throw std::runtime_error{std::string{"Function "} + fun_name +
                             " failed with cuda error code " +
                             std::to_string(error_code)};
  }
}

void copy(const void *src_device, size_t bytes, void *dst_host) {
  auto err = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    std::string msg{"Failed to copy data from device to host with error code "};
    msg += std::to_string(err);
    throw std::runtime_error{msg};
  }
}

template <typename float_t>
__device__ void impl_run_computation(
    const thrust::complex<float_t> *points,
    const thrust::complex<float_t> *parameters, int order, int rows, int cols,
    thrust::complex<float_t> r0c0, float_t r_unit, float_t c_unit,
    bool *dst_has_value, uint8_t *dst_nearest_index,
    thrust::complex<double> *dst_complex_diff, int iteration_times) {
  const auto global_offset = blockDim.x * blockIdx.x + threadIdx.x;
  const auto r = global_offset / cols;
  const auto c = global_offset % cols;

  thrust::complex<float_t> z{r0c0};
  z.imag(r * r_unit + r0c0.imag());
  z.real(c * c_unit + r0c0.real());

  newton_equation_base::single_result result;
  const bool ok = internal::compute_functions<
      float_t, thrust::complex<float_t>>::compute_single(parameters, points,
                                                         order, z,
                                                         iteration_times,
                                                         result);

  if (!ok) {
    result.nearest_point_idx = 255;
    result.difference.real(NAN);
    result.difference.imag(NAN);
  }
  dst_has_value[global_offset] = ok;
  dst_nearest_index[global_offset] = result.nearest_point_idx;
  dst_complex_diff[global_offset] = result.difference;
}

template <typename float_t>
__global__ void run_computation_shared_mem(
    const thrust::complex<float_t> *points_global,
    const thrust::complex<float_t> *parameters_global, int order, int rows,
    int cols, thrust::complex<float_t> r0c0, float_t r_unit, float_t c_unit,
    bool *dst_has_value, uint8_t *dst_nearest_index,
    thrust::complex<double> *dst_complex_diff, int iteration_times) {
  const auto global_offset = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_offset >= rows * cols) {
    return;
  }

  ///__shared__ extern const thrust::complex<float_t> *points;
  //__shared__ extern const thrust::complex<float_t> *parameters;
  __shared__ extern uint8_t shared_mem[];

  auto *const points = reinterpret_cast<thrust::complex<float_t> *>(shared_mem);
  thrust::complex<float_t> *const parameters = points + order;

  if (threadIdx.x == 0) {
    for (int idx = 0; idx < order; idx++) {
      points[idx] = points_global[idx];
    }
    for (int idx = 0; idx < order; idx++) {
      parameters[idx] = parameters_global[idx];
    }
  }

  __syncthreads();

  impl_run_computation<float_t>(
      points, parameters, order, rows, cols, r0c0, r_unit, c_unit,
      dst_has_value, dst_nearest_index, dst_complex_diff, iteration_times);
}

template <typename float_t>
__constant__ thrust::complex<float_t> const_points[255];
template <typename float_t>
__constant__ thrust::complex<float_t> const_parameters[255];

__constant__ char const_chars[128];

template <typename float_t>
std::mutex const_memory_lock;

template <typename float_t>
__global__ void run_computation_const_mem(
    int order, int rows, int cols, thrust::complex<float_t> r0c0,
    float_t r_unit, float_t c_unit, bool *dst_has_value,
    uint8_t *dst_nearest_index, thrust::complex<double> *dst_complex_diff,
    int iteration_times) {
  const auto global_offset = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_offset >= rows * cols) {
    return;
  }
  impl_run_computation<float_t>(
      const_points<float_t>, const_parameters<float_t>, order, rows, cols, r0c0,
      r_unit, c_unit, dst_has_value, dst_nearest_index, dst_complex_diff,
      iteration_times);
}

}  // namespace internal

template <typename float_t>
class cuda_equation_impl : public cuda_computer<float_t> {
 private:
  thrust::device_vector<thrust::complex<float_t>> m_points_device;
  thrust::device_vector<thrust::complex<float_t>> m_parameters_device;

  thrust::device_vector<bool> m_has_value_device;
  thrust::device_vector<uint8_t> m_nearest_index_device;
  thrust::device_vector<thrust::complex<double>> m_complex_diff_device;

  void compute_cuda_constant_memory(
      const newton_equation_base &eq, const fractal_utils::wind_base &_wind,
      int iteration_times, newton_equation_base::compute_option &opt) & {
    {
      std::vector<thrust::complex<float_t>> temp_points, temp_parameters;
      temp_points.resize(eq.order());
      temp_parameters.resize(eq.order());

      for (int o = 0; o < eq.order(); o++) {
        temp_points[o] = eq.point_at(o);
        temp_parameters[o] = eq.parameter_at(o);
      }
      auto err = cudaMemcpyToSymbol(
          internal::const_points<float_t>, temp_points.data(),
          temp_points.size() * sizeof(thrust::complex<float_t>), 0,
          cudaMemcpyKind::cudaMemcpyHostToDevice);
      internal::handle_error("cudaMemcpyToSymbol", err);
      err = cudaMemcpyToSymbol(
          internal::const_parameters<float_t>, temp_parameters.data(),
          temp_parameters.size() * sizeof(thrust::complex<float_t>), 0,
          cudaMemcpyKind::cudaMemcpyHostToDevice);
      internal::handle_error("cudaMemcpyToSymbol", err);
    }

    const size_t rows = opt.bool_has_result.rows();

    const size_t cols = opt.bool_has_result.cols();

    const auto &wind =
        dynamic_cast<const fractal_utils::center_wind<real_t> &>(_wind);

    const auto left_top_corner = wind.left_top_corner();
    const complex_t r0c0{left_top_corner[0], left_top_corner[1]};

    const real_t r_unit = -wind.y_span / rows;
    const real_t c_unit = wind.x_span / cols;

    const auto num_pixels = opt.bool_has_result.size();
    this->m_has_value_device.resize(num_pixels);
    this->m_nearest_index_device.resize(num_pixels);
    this->m_complex_diff_device.resize(num_pixels);

    constexpr size_t warp_size = 64;
    const size_t required_blocks = std::ceil(double(num_pixels) / warp_size);

    {
      internal::run_computation_const_mem<<<required_blocks, warp_size, 0>>>(
          eq.order(), opt.bool_has_result.rows(), opt.bool_has_result.cols(),
          r0c0, r_unit, c_unit, this->m_has_value_device.data().get(),
          this->m_nearest_index_device.data().get(),
          this->m_complex_diff_device.data().get(), iteration_times);
      auto err = cudaGetLastError();
      internal::handle_error("internal::run_computation_shared_mem", err);
    }

    {
      auto err = cudaDeviceSynchronize();
      internal::handle_error("cudaDeviceSynchronize", err);
    }

    internal::copy(this->m_has_value_device.data().get(),
                   opt.bool_has_result.bytes(), opt.bool_has_result.data());
    internal::copy(this->m_nearest_index_device.data().get(),
                   opt.u8_nearest_point_idx.bytes(),
                   opt.u8_nearest_point_idx.data());
    internal::copy(this->m_complex_diff_device.data().get(),
                   opt.f64complex_difference.bytes(),
                   opt.f64complex_difference.data());
  }

  void compute_cuda_shared_memory(const newton_equation_base &eq,
                                  const fractal_utils::wind_base &_wind,
                                  int iteration_times,
                                  newton_equation_base::compute_option &opt) & {
    this->m_points_device.resize(eq.order());
    this->m_parameters_device.resize(eq.order());
    for (int o = 0; o < eq.order(); o++) {
      this->m_points_device[o] = eq.point_at(o);
      this->m_parameters_device[o] = eq.parameter_at(o);
    }

    const size_t rows = opt.bool_has_result.rows();

    const size_t cols = opt.bool_has_result.cols();

    const auto &wind =
        dynamic_cast<const fractal_utils::center_wind<real_t> &>(_wind);

    const auto left_top_corner = wind.left_top_corner();
    const complex_t r0c0{left_top_corner[0], left_top_corner[1]};

    const real_t r_unit = -wind.y_span / rows;
    const real_t c_unit = wind.x_span / cols;

    const auto num_pixels = opt.bool_has_result.size();
    this->m_has_value_device.resize(num_pixels);
    this->m_nearest_index_device.resize(num_pixels);
    this->m_complex_diff_device.resize(num_pixels);

    constexpr size_t warp_size = 64;
    const size_t required_blocks = std::ceil(double(num_pixels) / warp_size);
    const size_t required_shared_mem =
        eq.order() * sizeof(thrust::complex<float_t>) * 2;

    {
      internal::run_computation_shared_mem<<<required_blocks, warp_size,
                                             required_shared_mem>>>(
          this->m_points_device.data().get(),
          this->m_parameters_device.data().get(), eq.order(),
          opt.bool_has_result.rows(), opt.bool_has_result.cols(), r0c0, r_unit,
          c_unit, this->m_has_value_device.data().get(),
          this->m_nearest_index_device.data().get(),
          this->m_complex_diff_device.data().get(), iteration_times);
      auto err = cudaGetLastError();
      internal::handle_error("internal::run_computation_shared_mem", err);
    }
    {
      auto err = cudaDeviceSynchronize();
      internal::handle_error("cudaDeviceSynchronize", err);
    }

    internal::copy(this->m_has_value_device.data().get(),
                   opt.bool_has_result.bytes(), opt.bool_has_result.data());
    internal::copy(this->m_nearest_index_device.data().get(),
                   opt.u8_nearest_point_idx.bytes(),
                   opt.u8_nearest_point_idx.data());
    internal::copy(this->m_complex_diff_device.data().get(),
                   opt.f64complex_difference.bytes(),
                   opt.f64complex_difference.data());
  }

  static bool check_constant_memory() {
    {
      int device = -10;
      auto err = cudaGetDevice(&device);
      if (err != cudaSuccess) {
        return false;
      }
      cudaDeviceProp prop{};
      err = cudaGetDeviceProperties(&prop, device);
      if (err != cudaSuccess) {
        return false;
      }
    }

    void *address = nullptr;
    {
      auto err = cudaGetSymbolAddress(&address, internal::const_chars);
      if (err != cudaSuccess) {
        return false;
      }
    }
    auto err =
        cudaGetSymbolAddress(&address, internal::const_parameters<float_t>);
    if (err != cudaSuccess) {
      return false;
    }
    err = cudaGetSymbolAddress(&address, internal::const_points<float_t>);
    if (err != cudaSuccess) {
      return false;
    }
    return true;
  }

 public:
  using real_t = float_t;
  using complex_t = thrust::complex<float_t>;
  void compute_cuda(const newton_equation_base &eq,
                    const fractal_utils::wind_base &_wind, int iteration_times,
                    newton_equation_base::compute_option &opt) &
      final {
    assert(opt.bool_has_result.rows() == opt.f64complex_difference.rows());
    assert(opt.f64complex_difference.rows() == opt.u8_nearest_point_idx.rows());
    assert(opt.bool_has_result.cols() == opt.f64complex_difference.cols());
    assert(opt.f64complex_difference.cols() == opt.u8_nearest_point_idx.cols());

    const bool is_constant_memory_ok = check_constant_memory();

    const bool use_const_memory =
        is_constant_memory_ok &&
        internal::const_memory_lock<float_t>.try_lock();

    if (use_const_memory) {
      // this->compute_cuda_shared_memory(eq, _wind, iteration_times, opt);
      this->compute_cuda_constant_memory(eq, _wind, iteration_times, opt);
    } else {
      this->compute_cuda_shared_memory(eq, _wind, iteration_times, opt);
    }

    if (use_const_memory) {
      internal::const_memory_lock<float_t>.unlock();
    }
  }
};

template <typename float_t>
tl::expected<std::unique_ptr<cuda_computer<float_t>>, std::string>
cuda_computer<float_t>::create() noexcept {
  int dev_num{-1};
  auto err = cudaGetDeviceCount(&dev_num);
  if (err != cudaSuccess) {
    return tl::make_unexpected(
        std::string{"Can not get the number of devices, error code = "} +
        std::to_string(err));
  }
  if (dev_num <= 0) {
    return tl::make_unexpected(std::string{"No cuda device."});
  }

  return std::unique_ptr<cuda_computer<float_t>>(
      new cuda_equation_impl<float_t>);
}

namespace internal {
template <>
[[nodiscard]] tl::expected<std::unique_ptr<cuda_computer<float>>, std::string>
create_computer<float>() noexcept {
  return cuda_computer<float>::create();
}

template <>
[[nodiscard]] tl::expected<std::unique_ptr<cuda_computer<double>>, std::string>
create_computer<double>() noexcept {
  return cuda_computer<double>::create();
}
}  // namespace internal

}  // namespace newton_fractal