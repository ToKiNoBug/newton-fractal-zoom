//
// Created by David on 2023/7/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_OPENCL_SUPPORT_H
#define NEWTON_FRACTAL_ZOOM_OPENCL_SUPPORT_H

#include "newton_equation_base.h"
#include <vector>
#include <string>
#include "object_creator.h"

namespace newton_fractal {

template <typename float_t>
[[nodiscard]] tl::expected<std::unique_ptr<newton_equation_base>, std::string>
create_opencl_equation(const opencl_option_t& option,
                       std::span<const std::complex<float_t>> points) noexcept;

[[nodiscard]] std::vector<std::string> opencl_platforms() noexcept;
[[nodiscard]] std::vector<std::string> opencl_devices(
    size_t platform_index) noexcept;

}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_OPENCL_SUPPORT_H
