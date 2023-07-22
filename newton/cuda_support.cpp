#include "cuda_support.cuh"
#include "newton_equation.hpp"
#include <fmt/format.h>

namespace newton_fractal {

template <typename float_t>
class cuda_equation : public newton_equation<std::complex<float_t>, float_t> {
 private:
  std::unique_ptr<cuda_computer<float_t>> m_computer;

  void initialize_computer() & {
    auto temp = internal::create_computer<float_t>();
    if (!temp) {
      throw std::runtime_error{fmt::format(
          "Failed to create cuda_computer, detail: {}", temp.error())};
    }
    this->m_computer = std::move(temp.value());
  }

 public:
  explicit cuda_equation(std::span<const std::complex<float_t>> points)
      : newton_equation<std::complex<float_t>, float_t>{points} {
    this->initialize_computer();
  }
  cuda_equation(cuda_equation&&) noexcept = default;

  cuda_equation(const cuda_equation& src)
      : newton_equation<std::complex<float_t>, float_t>(src) {
    this->initialize_computer();
  }

  [[nodiscard]] std::unique_ptr<newton_equation_base> copy()
      const noexcept final {
    return std::make_unique<cuda_equation<float_t>>(*this);
  }

  [[nodiscard]] static consteval bool is_fixed_precision() noexcept {
    return true;
  }
  [[nodiscard]] int precision() const noexcept {
    if constexpr (std::is_same_v<float_t, float>) {
      return 1;
    }
    return 2;
  }

  void compute(
      const fractal_utils::wind_base& _wind, int iteration_times,
      newton_equation_base::compute_option& opt) const noexcept override {
    this->m_computer->compute_cuda(*this, _wind, iteration_times, opt);
  }
};

template <typename float_t>
tl::expected<std::unique_ptr<newton_equation_base>, std::string>
cuda_computer<float_t>::create_complete_equation(
    std::span<const std::complex<float_t>> points) noexcept {
  try {
    return std::make_unique<cuda_equation<float_t>>(points);
  } catch (std::exception& e) {
    return tl::make_unexpected(fmt::format("Exception occurred: {}", e.what()));
  } catch (...) {
    return tl::make_unexpected("Unknown exception caught");
  }
}

namespace internal {
template <>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_complete_equation<float>(
    std::span<const std::complex<float_t>> points) noexcept {
  return cuda_computer<float>::create_complete_equation(points);
}

template <>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_complete_equation<double>(
    std::span<const std::complex<double>> points) noexcept {
  return cuda_computer<double>::create_complete_equation(points);
}
}  // namespace internal
}  // namespace newton_fractal