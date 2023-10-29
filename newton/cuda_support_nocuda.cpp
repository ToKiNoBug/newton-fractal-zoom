#include "cuda_support.cuh"

namespace newton_fractal {

namespace internal {
const char cuda_fail_msg[] = "CUDA support is disabled.";
template <>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_complete_equation<float>(
    std::span<const std::complex<float_t>>) noexcept {
  return tl::make_unexpected(cuda_fail_msg);
}

template <>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_complete_equation<double>(
    std::span<const std::complex<double>>) noexcept {
  return tl::make_unexpected(cuda_fail_msg);
}
}  // namespace internal

}  // namespace newton_fractal