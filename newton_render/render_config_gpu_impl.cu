#include "gpu_interface.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/allocate_unique.h>
#include "render_config.h"

namespace newton_fractal {

class render_config_gpu_impl : public render_config_gpu_interface {
 private:
  thrust::device_vector<render_config::render_method> m_methods;
  fractal_utils::pixel_RGB m_color_for_nan{0, 0, 0};

 public:
  [[nodiscard]] const render_config::render_method *method_ptr()
      const noexcept final {
    return this->m_methods.data().get();
  }
  [[nodiscard]] int num_methods() const noexcept final {
    return (int)this->m_methods.size();
  }

  [[nodiscard]] fractal_utils::pixel_RGB color_for_nan() const noexcept final {
    return this->m_color_for_nan;
  }

  [[nodiscard]] tl::expected<render_config, std::string> config()
      const noexcept final {
    if (this->m_methods.empty()) {
      return tl::make_unexpected("Render config not set.");
    }
    render_config ret;
    ret.methods.resize(this->m_methods.size());
    auto err = cudaMemcpy(
        ret.methods.data(), this->m_methods.data().get(),
        this->m_methods.size() * sizeof(render_config::render_method),
        cudaMemcpyKind::cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
      return tl::make_unexpected(
          std::string{"cudaMemcpy failed with error code "} +
          std::to_string(err));
    }
    ret.color_for_nan = this->m_color_for_nan;
    return ret;
  }

  [[nodiscard]] tl::expected<void, std::string> set_config(
      const render_config &rc) & noexcept final {
    try {
      this->m_methods = rc.methods;
      this->m_color_for_nan = rc.color_for_nan;
    } catch (std::exception &e) {
      return tl::make_unexpected(e.what());
    }
    return {};
  }
};

[[nodiscard]] tl::expected<std::unique_ptr<render_config_gpu_interface>,
                           std::string>
render_config_gpu_interface::create() noexcept {
  return std::make_unique<render_config_gpu_impl>();
}

}  // namespace newton_fractal