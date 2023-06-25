#include "run_compute.h"
#include <newton_archive.h>
#include <newton_fractal.h>
#include <fmt/format.h>
#include <newton_render.h>

tl::expected<std::pair<nf::render_config,
                       std::unique_ptr<nf::render_config_gpu_interface>>,
             std::string>

create_render_config_objects(const render_task& rt,
                             int archive_points) noexcept {
  nf::render_config render_config;
  {
    auto temp = nf::load_render_config(rt.render_config_filename);
    if (!temp.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to load render config from \"{}\", detail: {}.",
                      rt.render_config_filename, temp.error()));
    }
    render_config = std::move(temp.value());
  }
  if (render_config.methods.size() < archive_points) {
    return tl::make_unexpected(
        fmt::format("The archive metainfo shows that it contains {} points, "
                    "but the render config is available for only {} points.",
                    render_config.methods.size(), archive_points));
  }

  std::unique_ptr<nf::render_config_gpu_interface> render_config_gpu{nullptr};
  {
    auto temp = nf::render_config_gpu_interface::create();
    if (!temp.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to create render config gpu instance, detail: {}",
                      temp.error()));
    }
    render_config_gpu = std::move(temp.value());
  }
  {
    auto temp = render_config_gpu->set_config(render_config);
    if (!temp.has_value()) {
      return tl::make_unexpected(fmt::format(
          "Failed to pass render config to gpu, detail: {}", temp.error()));
    }
  }

  return std::make_pair(std::move(render_config), std::move(render_config_gpu));
}

tl::expected<void, std::string> run_render(const render_task& rt) noexcept {
  std::optional<nf::newton_archive> archive{std::nullopt};
  const nf::newton_archive* src{nullptr};
  if (!rt.archive_file.has_value()) {
    if (rt.archive_value != nullptr) {
      src = rt.archive_value;
    }
  } else {
    auto temp =
        nf::newton_archive::load_archive(rt.archive_file.value(), true, {});

    if (!temp.has_value()) {
      return tl::make_unexpected(fmt::format("Failed to load {}, detail: {}",
                                             rt.archive_file.value(),
                                             temp.error()));
    }
    archive = std::move(temp.value());
    src = &archive.value();
  }
  if (src == nullptr) {
    return tl::make_unexpected(
        "No value for source. Please assign the source filename, or run a "
        "computation together.");
  }

  auto render_option_objects =
      create_render_config_objects(rt, src->info().num_points());

  return {};
}