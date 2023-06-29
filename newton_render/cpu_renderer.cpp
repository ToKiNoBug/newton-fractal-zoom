
#include <complex>
#include "cpu_renderer.h"
#include "gpu_interface.h"

namespace newton_fractal {

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

  const int rows = map_has_value.rows();
  const int cols = map_has_value.cols();

  this->m_map_norm_arg.reset(rows, cols, sizeof(std::complex<double>));

  // compute norm and arg
  for (size_t i = 0; i < map_complex_difference.size(); i++) {
    auto cplx = map_complex_difference.at<std::complex<double>>(i);
    const double norm = std::abs(cplx);
    const double arg = std::arg(cplx);
    this->m_map_norm_arg.at<std::complex<double>>(i) = {norm, arg};
  }

  normalize_option mag_opt{INFINITY, -INFINITY}, arg_opt{INFINITY, -INFINITY};

  {
    int64_t processed_count = 0;
    for (int r = skip_rows; r < rows - skip_rows; r++) {
      for (int c = skip_cols; c < cols - skip_cols; c++) {
        if (!map_has_value.at<bool>(r, c)) {
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

  for (int r = skip_rows; r < rows - skip_rows; r++) {
    for (int c = skip_cols; c < cols - skip_cols; c++) {
      const auto norm_arg = this->m_map_norm_arg.at<std::complex<double>>(r, c);
      const double mag = norm_arg.real();
      const double arg = norm_arg.imag();

      image_u8c3.at<fractal_utils::pixel_RGB>(r, c) =
          render_cpu(config, map_has_value.at<bool>(r, c),
                     map_nearest_idx.at<uint8_t>(r, c), arg_opt.normalize(mag),
                     arg_opt.normalize(arg));
    }
  }
}

}  // namespace newton_fractal