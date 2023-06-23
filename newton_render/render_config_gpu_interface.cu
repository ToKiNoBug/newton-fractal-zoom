#include "gpu_interface.h"
#include "gpu_internal.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>

namespace newton_fractal {

class render_config_gpu_implementation : public render_config_gpu_interface {
 private:
  int m_size{0};
  fractal_utils::pixel_RGB m_color_for_nan{0xFF000000};
  internal::unique_cu_ptr<render_config::render_method> m_methods{nullptr};
  cudaError_t m_error_code{cudaError_t::cudaSuccess};

 public:
  render_config_gpu_implementation();
  ~render_config_gpu_implementation() override = default;

  [[nodiscard]] int size() const noexcept { return this->m_size; }
  [[nodiscard]] auto color_for_nan() const noexcept {
    return this->m_color_for_nan;
  }

  [[nodiscard]] int error_code() const noexcept final {
    return this->m_error_code;
  }
  [[nodiscard]] bool ok() const noexcept final {
    return this->error_code() == cudaError_t::cudaSuccess;
  }

 public:
  static constexpr size_t gpu_memory_capacity = 255;

  [[nodiscard]] tl::expected<void, std::string> set_config(
      const render_config &rc) & noexcept override;
  [[nodiscard]] tl::expected<render_config, std::string> config()
      const noexcept override;

  [[nodiscard]] const render_config::render_method *method_ptr()
      const noexcept override {
    return this->m_methods.get();
  }
  [[nodiscard]] int num_methods() const noexcept override {
    return this->size();
  }
};

render_config_gpu_implementation::render_config_gpu_implementation() {
  auto temp =
      allocate_device_memory<render_config::render_method>(gpu_memory_capacity);
  if (!temp) {
    this->m_error_code = temp.error();
  } else {
    this->m_methods = std::move(temp.value());
  }
}

tl::expected<void, std::string> render_config_gpu_implementation::set_config(
    const render_config &rc) & noexcept {
  if (rc.methods.empty() || rc.methods.size() <= gpu_memory_capacity) {
    return tl::make_unexpected("Invalid number of render_methods.");
  }

  auto err =
      cudaMemcpy(this->m_methods.get(), rc.methods.data(),
                 rc.methods.size() * sizeof(render_config::render_method),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
  if (err != cudaError_t::cudaSuccess) {
    return tl::make_unexpected("cudaMemcpy failed.");
  }

  this->m_size = (int)rc.methods.size();
  this->m_color_for_nan = rc.color_for_nan;
  return {};
}

[[nodiscard]] tl::expected<render_config, std::string>
render_config_gpu_implementation::config() const noexcept {
  if (!this->ok()) {
    return tl::make_unexpected(
        "The render_config_gpu_interface instance is not ok. cuda error code "
        "= " +
        std::to_string(this->error_code()));
  }

  render_config ret;

  ret.color_for_nan = this->m_color_for_nan;
  ret.methods.resize(this->m_size);

  auto err = cudaMemcpy(ret.methods.data(), this->m_methods.get(),
                        this->m_size * sizeof(render_config::render_method),
                        cudaMemcpyKind::cudaMemcpyDeviceToHost);
  if (err != cudaError_t::cudaSuccess) {
    return tl::make_unexpected("cudaMemcpy failed.");
  }

  return ret;
}

tl::expected<std::unique_ptr<render_config_gpu_interface>, std::string>
render_config_gpu_interface::create() noexcept {
  auto ret = std::make_unique<render_config_gpu_implementation>();
  if (!ret->ok()) {
    return tl::make_unexpected(
        "The instance initialization failed with cuda error code " +
        std::to_string(ret->error_code()));
  }

  return ret;
}

}  // namespace newton_fractal