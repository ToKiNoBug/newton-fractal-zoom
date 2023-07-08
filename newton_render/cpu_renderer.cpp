
#include <complex>
#include "cpu_renderer.h"
#include "gpu_interface.h"
#include <fmt/format.h>

namespace newton_fractal {

namespace internal {

void set_data(fractal_utils::constant_view src,
              std::optional<std::variant<fractal_utils::unique_map,
                                         fractal_utils::constant_view>> &dst,
              bool deep_copy) noexcept {
  if (deep_copy) {
    if (!dst.has_value() || dst.value().index() != 0) {
      dst.emplace(fractal_utils::unique_map{});
    }
    auto &umap = std::get<0>(dst.value());
    umap.reset(src.rows(), src.cols(), src.element_bytes());
    memcpy(umap.data(), src.data(), src.bytes());
  } else {
    if (!dst.has_value() || dst.value().index() != 1) {
      dst.reset();
    }
    dst.emplace(fractal_utils::constant_view{src});
  }
}

fractal_utils::constant_view get_map(
    const std::optional<
        std::variant<fractal_utils::unique_map, fractal_utils::constant_view>>
        &src) noexcept {
  return std::visit(
      [](const auto &val) -> fractal_utils::constant_view {
        return fractal_utils::constant_view{val};
      },
      src.value());
}

[[nodiscard]] uint8_t compute_max_nearest_index(
    fractal_utils::constant_view has_value,
    fractal_utils::constant_view nearest_index) noexcept {
  assert(has_value.rows() == nearest_index.rows());
  assert(has_value.cols() == nearest_index.cols());

  uint8_t max{0};
  for (size_t idx = 0; idx < nearest_index.size(); idx++) {
    if (!has_value.at<uint8_t>(idx)) {
      continue;
    }
    max = std::max(max, nearest_index.at<uint8_t>(idx));
  }
  return max;
}

}  // namespace internal

void cpu_renderer::set_data(fractal_utils::constant_view has_value,
                            fractal_utils::constant_view nearest_idx,
                            fractal_utils::constant_view complex_difference,
                            bool deep_copy) & noexcept {
  assert(has_value.rows() == nearest_idx.rows());
  assert(nearest_idx.rows() == complex_difference.rows());
  assert(has_value.cols() == nearest_idx.cols());
  assert(nearest_idx.cols() == complex_difference.cols());
  internal::set_data(has_value, this->m_has_value, deep_copy);
  internal::set_data(nearest_idx, this->m_nearest_idx, deep_copy);
  internal::set_data(complex_difference, this->m_complex_difference, deep_copy);
  this->compute_norm_arg();
  this->compute_nearest_idx_max();
}
void cpu_renderer::set_data(
    fractal_utils::unique_map &&has_value,
    fractal_utils::unique_map &&nearest_idx,
    fractal_utils::unique_map &&complex_difference) & noexcept {
  assert(has_value.rows() == nearest_idx.rows());
  assert(nearest_idx.rows() == complex_difference.rows());
  assert(has_value.cols() == nearest_idx.cols());
  assert(nearest_idx.cols() == complex_difference.cols());
  this->m_has_value = has_value;
  this->m_nearest_idx = nearest_idx;
  this->m_complex_difference = complex_difference;
  this->compute_norm_arg();
  this->compute_nearest_idx_max();
}

void cpu_renderer::compute_nearest_idx_max() & noexcept {
  this->m_nearest_idx_max = internal::compute_max_nearest_index(
      internal::get_map(this->m_has_value),
      internal::get_map(this->m_nearest_idx));
}

void cpu_renderer::compute_norm_arg() & noexcept {
  this->m_map_norm_arg.reset(this->rows(), this->cols(),
                             sizeof(std::complex<double>));

  for (size_t i = 0; i < internal::get_map(this->m_complex_difference).size();
       i++) {
    if (!internal::get_map(this->m_has_value).at<bool>(i)) {
      continue;
    }
    auto cplx = internal::get_map(this->m_complex_difference)
                    .at<std::complex<double>>(i);
    const double norm = std::abs(cplx);
    const double arg = std::arg(cplx);
    this->m_map_norm_arg.at<std::complex<double>>(i) = {norm, arg};
  }
}

void cpu_renderer::reset() & noexcept {
  this->m_has_value.reset();
  this->m_nearest_idx.reset();
  this->m_complex_difference.reset();
  this->m_nearest_idx_max.reset();
}

tl::expected<void, std::string> cpu_renderer::render(
    const render_config &config, fractal_utils::map_view image_u8c3,
    int skip_rows, int skip_cols) const noexcept {
  if (this->m_nearest_idx_max.value() >= config.methods.size()) {
    return tl::make_unexpected(
        fmt::format("The render config contains {} colors, but required {}.",
                    config.methods.size(), this->m_nearest_idx_max.value()));
  }

  if (this->m_map_norm_arg.rows() != image_u8c3.rows() ||
      this->m_map_norm_arg.cols() != image_u8c3.cols()) {
    return tl::make_unexpected(
        "The size of this->m_map_norm_arg and this->m_map_norm_arg mismatch.");
  }

  normalize_option mag_opt{INFINITY, -INFINITY}, arg_opt{INFINITY, -INFINITY};
  {
    int64_t processed_count = 0;
    for (int r = skip_rows; r < this->rows() - skip_rows; r++) {
      for (int c = skip_cols; c < this->cols() - skip_cols; c++) {
        if (!internal::get_map(m_has_value).at<bool>(r, c)) {
          continue;
        }
        const auto norm_arg =
            this->m_map_norm_arg.at<std::complex<double>>(r, c);
        const double mag = norm_arg.real();
        const double arg = norm_arg.imag();

        mag_opt.add_data(mag);
        arg_opt.add_data(arg);
        processed_count++;
      }
    }
    if (processed_count <= 0) {
      mag_opt.min = 0;
      arg_opt.min = 0;
    }
    mag_opt.max = std::max(mag_opt.max, std::nextafter(mag_opt.min, 1));
    arg_opt.max = std::max(arg_opt.max, std::nextafter(arg_opt.min, 1));
  }

  for (int r = skip_rows; r < this->rows() - skip_rows; r++) {
    for (int c = skip_cols; c < this->cols() - skip_cols; c++) {
      const auto norm_arg = this->m_map_norm_arg.at<std::complex<double>>(r, c);
      const double mag = norm_arg.real();
      const double arg = norm_arg.imag();

      image_u8c3.at<fractal_utils::pixel_RGB>(r, c) = render_cpu(
          config, internal::get_map(this->m_has_value).at<bool>(r, c),
          internal::get_map(m_nearest_idx).at<uint8_t>(r, c),
          (float)mag_opt.normalize(mag), (float)arg_opt.normalize(arg));
    }
  }

  return {};
}

void cpu_renderer::render(const newton_fractal::render_config &config,
                          fractal_utils::constant_view map_has_value,
                          fractal_utils::constant_view map_nearest_idx,
                          fractal_utils::constant_view map_complex_difference,
                          fractal_utils::map_view image_u8c3, int skip_rows,
                          int skip_cols) & noexcept {
  assert(map_has_value.rows() == map_nearest_idx.rows());
  assert(map_nearest_idx.rows() == map_complex_difference.rows());
  assert(map_complex_difference.rows() == image_u8c3.rows());

  assert(map_has_value.cols() == map_nearest_idx.cols());
  assert(map_nearest_idx.cols() == map_complex_difference.cols());
  assert(map_complex_difference.cols() == image_u8c3.cols());

  this->set_data(map_has_value, map_nearest_idx, map_complex_difference, false);
  auto err = this->render(config, image_u8c3, skip_rows, skip_cols);
  if (!err) {
    fmt::print("Failed to render because {}", err.error());
  }
}

}  // namespace newton_fractal