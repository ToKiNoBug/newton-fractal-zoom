//
// Created by David on 2023/7/23.
//
#ifdef NF_HAS_OPENCL_HPP
#include <CL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <fmt/format.h>
#include <tl/expected.hpp>
#include "newton_equation_base.h"
#include "newton_equation.hpp"
#include "OpenCL_support.h"

extern "C" {
extern const unsigned char newton_fractal_computation_cl[];
extern const unsigned int newton_fractal_computation_cl_length;
}

namespace newton_fractal {

[[nodiscard]] tl::expected<std::vector<cl_platform_id>, cl_int>
get_all_platforms() noexcept;

void initialize_buffer(const cl::Context& context, cl::Buffer& buf,
                       size_t bytes, cl_mem_flags flags) noexcept(false);
void resize_buffer(cl::Buffer& buf, size_t bytes) noexcept(false);

void resize_dest_buffer(size_t num_pixels, cl::Buffer& buf_has_value,
                        cl::Buffer& buf_nearest_index,
                        cl::Buffer& buf_complex_diff) noexcept;

struct basic_objects {
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  cl::Kernel kernel;

  bool is_double_disabled{false};

  struct options {
    bool disable_float{false};
    bool disable_double{false};
  };

  tl::expected<void, std::string> initialize(size_t platform_index,
                                             size_t device_index,
                                             std::string_view source_code,
                                             const char* fun_name,
                                             const options& opt) & noexcept;
};

template <int precision>
class opencl_equation : public equation_fixed_prec<precision> {
 public:
  static_assert(precision == 1 || precision == 2);
  using base_t = equation_fixed_prec<precision>;
  using real_type = typename base_t::real_type;
  using complex_type = typename base_t::complex_type;

 private:
  basic_objects m_basic_objs;

  cl::Buffer m_buf_points;
  cl::Buffer m_buf_parameters;

  cl::Buffer m_buf_has_value;
  cl::Buffer m_buf_nearest_index;
  cl::Buffer m_buf_complex_diff;

 public:
  opencl_equation(size_t platform_index, size_t device_index,
                  std::span<const complex_type> points);

  static tl::expected<std::unique_ptr<opencl_equation>, std::string> create(
      size_t platform_index, size_t device_index,
      std::span<const complex_type> points) noexcept;

  void run_computation(
      const fractal_utils::wind_base& _wind, int iteration_times,
      typename newton_equation_base::compute_option& opt) & noexcept(false);
};

#define handle_error_expected(fun_name, error_code)                            \
  if (error_code != CL_SUCCESS) {                                              \
    return tl::make_unexpected(                                                \
        fmt::format("Function {} failed with opencl error code {}", #fun_name, \
                    error_code));                                              \
  }

#define handle_error_throw(fun_name, error_code)                               \
  if (error_code != CL_SUCCESS) {                                              \
    throw std::runtime_error{                                                  \
        fmt::format("Function {} failed with opencl error code {}", #fun_name, \
                    error_code)};                                              \
  }

tl::expected<void, std::string> basic_objects::initialize(
    size_t platform_index, size_t device_index, std::string_view source_code,
    const char* fun_name, const options& opt) & noexcept {
  {
    cl_platform_id plats[1024];
    cl_uint num_plats{0};
    auto err = clGetPlatformIDs(1024, plats, &num_plats);
    handle_error_expected(clGetPlatformIDs, err);
    if (platform_index >= num_plats) {
      return tl::make_unexpected(
          fmt::format("Invalid index of opencl platform, there are {} "
                      "platform(s), but assigned the {}-th.",
                      num_plats, platform_index));
    }
    this->platform = cl::Platform{plats[platform_index]};
    fmt::print("The opencl platform is: {}\n",
               this->platform.getInfo<CL_PLATFORM_NAME>());
  }

  {
    std::vector<cl::Device> devices;
    auto err = this->platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    handle_error_expected(cl::Platform::getDevices, err);
    if (device_index >= devices.size()) {
      return tl::make_unexpected(
          fmt::format("Invalid index of opencl device, there are {} "
                      "devices(s), but assigned the {}-th.",
                      devices.size(), device_index));
    }
    this->device = devices[device_index];
    fmt::print("The opencl device is: {}\n",
               this->device.getInfo<CL_DEVICE_NAME>());
  }

  cl_int err;
  {
    this->context = cl::Context{this->device, nullptr, nullptr, nullptr, &err};
    handle_error_expected(
        cl::Context::Context(
            const Device& device, cl_context_properties* properties = 0,
            void (*)(const char*, const void*, ::size_t, void*) notifyFptr = 0,
            void* data = 0, cl_int* err = 0),
        err);
  }

  {
#if NF_HAS_OPENCL_HPP
    this->queue = cl::CommandQueue{this->context, this->device,
                                   cl::QueueProperties::None, &err};
    handle_error_expected(
        cl::CommandQueue::CommandQueue(
            const Context& context, const Device& device,
            QueueProperties properties, cl_int* err = nullptr),
        err);
#else
    this->queue =
        cl::CommandQueue{this->context, {this->device}, nullptr, &err};
    handle_error_expected(
        cl::CommandQueue::CommandQueue(
            const Context& context, const Device& device,
            const cl_queue_properties* properties = 0, cl_int* err = 0),
        err);
#endif
  }

  {
    this->program = cl::Program{
        this->context,
        cl::Program::Sources{{source_code.data(), source_code.length()}}, &err};
    handle_error_expected(
        cl::Program::Program(const Context& context, const Sources& sources,
                             cl_int* err = NULL),
        err);

    std::string build_args{""};
    if (opt.disable_float) {
      build_args += " -D NF_OPENCL_DISABLE_FP32";
    }
    if (opt.disable_double) {
      build_args += " -D NF_OPENCL_DISABLE_FP64";
    }

    err = this->program.build(this->device, build_args.c_str());
    if (err != CL_SUCCESS) {
      cl_int err_info;
      std::string info = this->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
          this->device, &err_info);
      if (err_info != CL_SUCCESS) {
        info = fmt::format(
            "Failed to get build log, function "
            "cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG> failed with "
            "opencl error code {}",
            err_info);
      }
      return tl::make_unexpected(fmt::format(
          "Failed to build opencl program (error code {}), build log:\noe{}",
          err, info));
    }
  }

  {
    this->kernel = cl::Kernel{this->program, fun_name, &err};
    handle_error_expected(cl::Kernel::Kernel(const Program& program,
                                             const char* name, cl_int* err = 0),
                          err);
  }

  return {};
}

template <int precision>
opencl_equation<precision>::opencl_equation(
    size_t platform_index, size_t device_index,
    std::span<const complex_type> points)
    : base_t{points} {
  const char* fun_name = nullptr;
  if constexpr (precision == 1) {
    fun_name = "run_computation_float";
  } else {
    fun_name = "run_computation_double";
  }
  basic_objects::options option;
  if (precision == 1) {
    option.disable_double = true;
  } else {
    option.disable_float = true;
  }

  auto err =
      this->m_basic_objs.initialize(platform_index, device_index,
                                    {(const char*)newton_fractal_computation_cl,
                                     newton_fractal_computation_cl_length},
                                    fun_name, option);
  if (!err) {
    throw std::runtime_error{fmt::format(
        "Failed to initialize opencl computation objects, detail: {}",
        err.error())};
  }

  constexpr size_t points_capacity = 30;
  initialize_buffer(this->m_basic_objs.context, this->m_buf_points,
                    points_capacity * sizeof(complex_type),
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY);
  initialize_buffer(this->m_basic_objs.context, this->m_buf_parameters,
                    points_capacity * sizeof(complex_type),
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY);

  constexpr size_t pixel_capacity = 1;
  initialize_buffer(this->m_basic_objs.context, this->m_buf_has_value,
                    pixel_capacity * sizeof(bool),
                    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY);
  initialize_buffer(this->m_basic_objs.context, this->m_buf_parameters,
                    pixel_capacity * sizeof(uint8_t),
                    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY);
  initialize_buffer(this->m_basic_objs.context, this->m_buf_complex_diff,
                    pixel_capacity * sizeof(std::complex<double>),
                    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY);
}

void initialize_buffer(const cl::Context& context, cl::Buffer& buf,
                       size_t bytes, cl_mem_flags flags) noexcept(false) {
  cl_int error_code;
  buf = cl::Buffer{context, flags, bytes, nullptr, &error_code};
  handle_error_throw(
      cl::Buffer::Buffer(const Context& context, cl_mem_flags flags,
                         ::size_t size, void* host_ptr = 0, cl_int* err = 0),
      error_code);
}

void resize_buffer(cl::Buffer& buf, size_t bytes) noexcept(false) {
  cl_int error_code;
  const auto actual_bytes = buf.getInfo<CL_MEM_SIZE>(&error_code);
  handle_error_throw(cl::Buffer::getInfo<CL_MEM_SIZE>, error_code);

  if (actual_bytes >= bytes) {
    return;
  }

  const auto flags = buf.getInfo<CL_MEM_FLAGS>(&error_code);
  handle_error_throw(cl::Buffer::getInfo<CL_MEM_FLAGS>, error_code);

  const auto context = buf.getInfo<CL_MEM_CONTEXT>(&error_code);
  handle_error_throw(cl::Buffer::getInfo<CL_MEM_CONTEXT>, error_code);

  initialize_buffer(context, buf, bytes, flags);
}

void resize_dest_buffer(size_t num_pixels, cl::Buffer& buf_has_value,
                        cl::Buffer& buf_nearest_index,
                        cl::Buffer& buf_complex_diff) noexcept {
  resize_buffer(buf_has_value, num_pixels * sizeof(bool));
  resize_buffer(buf_nearest_index, num_pixels * sizeof(uint8_t));
  resize_buffer(buf_complex_diff, num_pixels * sizeof(std::complex<double>));
}

template <int precision>
void opencl_equation<precision>::run_computation(
    const fractal_utils::wind_base& _wind, int iteration_times,
    typename newton_equation_base::compute_option& opt) & noexcept(false) {
  assert(opt.bool_has_result.rows() == opt.f64complex_difference.rows());
  assert(opt.f64complex_difference.rows() == opt.u8_nearest_point_idx.rows());
  const size_t rows = opt.bool_has_result.rows();

  assert(opt.bool_has_result.cols() == opt.f64complex_difference.cols());
  assert(opt.f64complex_difference.cols() == opt.u8_nearest_point_idx.cols());
  const size_t cols = opt.bool_has_result.cols();

  const auto& wind =
      dynamic_cast<const fractal_utils::center_wind<real_type>&>(_wind);

  const auto left_top_corner = wind.left_top_corner();
  const complex_type r0c0{left_top_corner[0], left_top_corner[1]};

  const real_type r_unit = -wind.y_span / rows;
  const real_type c_unit = wind.x_span / cols;

  auto fun_set_points_parameters = [this](std::span<complex_type> host,
                                          cl::Buffer& device_buf) {
    resize_buffer(device_buf, host.size_bytes());

    auto err = this->m_basic_objs.queue.enqueueWriteBuffer(
        device_buf, true, 0, host.size_bytes(), host.data());
    handle_error_throw(cl::CommandQueue::enqueueWriteBuffer, err);
  };

  fun_set_points_parameters(this->_points, this->m_buf_points);
  fun_set_points_parameters(this->_parameters, this->m_buf_parameters);

  resize_dest_buffer(rows * cols, this->m_buf_has_value,
                     this->m_buf_nearest_index, this->m_buf_complex_diff);

  cl_int err;
  {
    cl_uint arg_index = 0;
    err = this->m_basic_objs.kernel.setArg(arg_index++, this->m_buf_points);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.setArg(arg_index++, this->m_buf_parameters);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.template setArg<int>(arg_index++, rows);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.template setArg<int>(arg_index++, cols);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.setArg(arg_index++, r0c0);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.setArg(arg_index++, r_unit);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.setArg(arg_index++, c_unit);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.setArg(arg_index++, this->m_buf_has_value);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.setArg(arg_index++,
                                           this->m_buf_nearest_index);
    handle_error_throw(cl::Kernel::setArg, err);
    err =
        this->m_basic_objs.kernel.setArg(arg_index++, this->m_buf_complex_diff);
    handle_error_throw(cl::Kernel::setArg, err);
    err = this->m_basic_objs.kernel.template setArg<int>(arg_index++,
                                                         iteration_times);
    handle_error_throw(cl::Kernel::setArg, err);
  }

  constexpr size_t block_size = 64;
  const size_t required_blocks = std::ceil(double(rows * cols) / block_size);
  err = this->m_basic_objs.queue.enqueueNDRangeKernel(
      this->m_basic_objs.kernel, cl::NDRange{0},
      cl::NDRange{required_blocks * block_size}, cl::NDRange{block_size});
  handle_error_throw(cl::CommandQueue::enqueueNDRangeKernel, err);

  auto fun_read_matrix = [this](const cl::Buffer& src, fu::map_view dst) {
    const auto err = this->m_basic_objs.queue.enqueueReadBuffer(
        src, false, 0, dst.bytes(), dst.data());
    handle_error_throw(cl::CommandQueue::enqueueReadBuffer, err);
  };

  fun_read_matrix(opt.bool_has_result, this->m_buf_has_value);
  fun_read_matrix(opt.u8_nearest_point_idx, this->m_buf_nearest_index);
  fun_read_matrix(opt.f64complex_difference, this->m_buf_complex_diff);

  err = this->m_basic_objs.queue.finish();
  handle_error_throw(cl::CommandQueue::finish(), err);
}

template <int precision>

tl::expected<std::unique_ptr<opencl_equation<precision>>, std::string>
opencl_equation<precision>::create(
    size_t platform_index, size_t device_index,
    std::span<const complex_type> points) noexcept {
  try {
    auto ret = std::make_unique<opencl_equation<precision>>(
        platform_index, device_index, points);
    return ret;
  } catch (std::exception& e) {
    return tl::make_unexpected(
        fmt::format("Exception occurred, detail: {}.", e.what()));
  } catch (...) {
    return tl::make_unexpected("Unknown exception occurred.");
  }
}

template <>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_opencl_equation<float>(
    size_t platform_index, size_t device_index,
    std::span<const std::complex<float>> points) noexcept {
  auto exp = opencl_equation<1>::create(platform_index, device_index, points);
  return exp;
}

template <>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_opencl_equation<double>(
    size_t platform_index, size_t device_index,
    std::span<const std::complex<double>> points) noexcept {
  auto exp = opencl_equation<2>::create(platform_index, device_index, points);
  return exp;
}

[[nodiscard]] tl::expected<std::vector<cl_platform_id>, cl_int>
get_all_platforms() noexcept {
  cl_int err;
  cl_uint num_plats;
  err = clGetPlatformIDs(0, nullptr, &num_plats);
  if (err != CL_SUCCESS) return tl::make_unexpected(err);

  std::vector<cl_platform_id> ret;
  ret.resize(num_plats);
  err = clGetPlatformIDs(ret.size(), ret.data(), nullptr);
  if (err != CL_SUCCESS) return tl::make_unexpected(err);
  return ret;
}

[[nodiscard]] std::vector<std::string> opencl_platforms() noexcept {
  auto plats_exp = get_all_platforms();
  if (!plats_exp.has_value()) {
    return {};
  }
  auto& plats = plats_exp.value();
  const auto num_plats = plats.size();

  std::vector<std::string> ret;
  ret.resize(num_plats);
  for (size_t pid = 0; pid < num_plats; pid++) {
    cl_int err_code;
    cl::Platform plat{plats[pid]};
    ret[pid] = plat.getInfo<CL_PLATFORM_NAME>(&err_code);
    if (err_code != CL_SUCCESS) {
      ret[pid] = fmt::format(
          "Failed to get platform name, "
          "cl::Platform::getInfo<CL_PLATFORM_NAME> failed with opencl error "
          "code {}",
          err_code);
    }
  }
  return ret;
}

[[nodiscard]] std::vector<std::string> opencl_devices(
    size_t platform_index) noexcept {
  cl_platform_id plat_c{nullptr};
  try {
    plat_c = get_all_platforms().value()[platform_index];
  } catch (...) {
    return {};
  }
  cl::Platform plat{plat_c};

  cl_int err;
  std::vector<cl::Device> devices;
  err = plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (err != CL_SUCCESS) {
    return {};
  }

  std::vector<std::string> ret;
  ret.reserve(devices.size());
  for (auto& device : devices) {
    std::string name = device.getInfo<CL_DEVICE_NAME>(&err);
    if (err != CL_SUCCESS) {
      name = fmt::format(
          "Failed to get device name, function "
          "cl::Device::getInfo<CL_DEVICE_NAME> failed with opencl error code "
          "{}",
          err);
    }
    ret.emplace_back(name);
  }
  return ret;
}
}  // namespace newton_fractal