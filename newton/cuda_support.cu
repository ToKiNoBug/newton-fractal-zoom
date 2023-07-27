//
// Created by David on 2023/7/22.
//
#define JSON_HAS_RANGES false

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <exception>
#include "computation.hpp"
#include "cuda_support.cuh"
// #include "newton_equation.hpp"

namespace newton_fractal {

namespace internal {

void copy(const void *src_device, size_t bytes, void *dst_host) {
  auto err = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    std::string msg{"Failed to copy data from device to host with error code "};
    msg += std::to_string(err);
    throw std::runtime_error{msg};
  }
}

template <typename float_t>
__global__ void run_computation(
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

  thrust::complex<float_t> *const points =
      reinterpret_cast<thrust::complex<float_t> *>(shared_mem);
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

}  // namespace internal

template <typename float_t>
class cuda_equation_impl : public cuda_computer<float_t> {
 private:
  thrust::device_vector<thrust::complex<float_t>> m_points_device;
  thrust::device_vector<thrust::complex<float_t>> m_parameters_device;

  thrust::device_vector<bool> m_has_value_device;
  thrust::device_vector<uint8_t> m_nearest_index_device;
  thrust::device_vector<thrust::complex<double>> m_complex_diff_device;

 public:
  using real_t = float_t;
  using complex_t = thrust::complex<float_t>;
  void compute_cuda(const newton_equation_base &eq,
                    const fractal_utils::wind_base &_wind, int iteration_times,
                    newton_equation_base::compute_option &opt) &
      final {
    this->m_points_device.resize(eq.order());
    this->m_parameters_device.resize(eq.order());

    for (int o = 0; o < eq.order(); o++) {
      this->m_points_device[o] = eq.point_at(o);
      this->m_parameters_device[o] = eq.parameter_at(o);
    }

    assert(opt.bool_has_result.rows() == opt.f64complex_difference.rows());
    assert(opt.f64complex_difference.rows() == opt.u8_nearest_point_idx.rows());
    const size_t rows = opt.bool_has_result.rows();

    assert(opt.bool_has_result.cols() == opt.f64complex_difference.cols());
    assert(opt.f64complex_difference.cols() == opt.u8_nearest_point_idx.cols());
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

    internal::
        run_computation<<<required_blocks, warp_size, required_shared_mem>>>(
            this->m_points_device.data().get(),
            this->m_parameters_device.data().get(), eq.order(),
            opt.bool_has_result.rows(), opt.bool_has_result.cols(), r0c0,
            r_unit, c_unit, this->m_has_value_device.data().get(),
            this->m_nearest_index_device.data().get(),
            this->m_complex_diff_device.data().get(), iteration_times);

    cudaDeviceSynchronize();

    internal::copy(this->m_has_value_device.data().get(),
                   opt.bool_has_result.bytes(), opt.bool_has_result.data());
    internal::copy(this->m_nearest_index_device.data().get(),
                   opt.u8_nearest_point_idx.bytes(),
                   opt.u8_nearest_point_idx.data());
    internal::copy(this->m_complex_diff_device.data().get(),
                   opt.f64complex_difference.bytes(),
                   opt.f64complex_difference.data());
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