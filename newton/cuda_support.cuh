//
// Created by David on 2023/7/22.
//

#ifndef NEWTON_FRACTAL_ZOOM_CUDA_SUPPORT_CUH
#define NEWTON_FRACTAL_ZOOM_CUDA_SUPPORT_CUH

#include "newton_equation_base.h"

namespace newton_fractal {
template <typename float_t>
class cuda_computer {
 private:
 protected:
 public:
  virtual ~cuda_computer() = default;
  [[nodiscard]] static tl::expected<std::unique_ptr<cuda_computer>, std::string>
  create() noexcept;

  [[nodiscard]] static tl::expected<std::unique_ptr<newton_equation_base>,
                                    std::string>
  create_complete_equation(
      std::span<const std::complex<float_t>> points) noexcept;

  virtual void compute_cuda(const newton_equation_base &eq,
                            const fractal_utils::wind_base &wind,
                            int iteration_times,
                            newton_equation_base::compute_option &opt) & = 0;
};

namespace internal {
template <typename float_t>
[[nodiscard]] tl::expected<std::unique_ptr<cuda_computer<float_t>>, std::string>
create_computer() noexcept;

template <typename float_t>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_complete_equation(
    std::span<const std::complex<float_t>> points) noexcept;
}  // namespace internal
}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_CUDA_SUPPORT_CUH
