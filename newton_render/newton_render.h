//
// Created by David on 2023/6/21.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_RENDER_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_RENDER_H

#include <memory>
#include "render_config.h"
#include "gpu_interface.h"
#include "cpu_renderer.h"
#include <nlohmann/json.hpp>

using njson = nlohmann::json;

namespace newton_fractal {

tl::expected<render_config, std::string> load_render_config(
    const njson&) noexcept;

njson save_render_config(const render_config& nc) noexcept;
}  // namespace newton_fractal

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_RENDER_H
