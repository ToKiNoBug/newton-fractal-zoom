#include "gpu_interface.h"

using namespace newton_fractal;

tl::expected<std::unique_ptr<render_config_gpu_interface>, std::string>
render_config_gpu_interface::create() noexcept {
  return tl::make_unexpected("CUDA support is disabled.");
}

tl::expected<std::unique_ptr<gpu_render>, std::string> gpu_render::create(
    int, int) noexcept {
  return tl::make_unexpected("CUDA support is disabled.");
}

tl::expected<void, std::string> gpu_render::render(
    const newton_fractal::render_config_gpu_interface &config,
    fractal_utils::map_view image_u8c3, int skip_rows,
    int skip_cols) & noexcept {
  return tl::make_unexpected("CUDA support is disabled.");
}