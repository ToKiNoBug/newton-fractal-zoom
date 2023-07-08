#include <iostream>
#include <newton_archive.h>
#include <newton_fractal.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <newton_render.h>
#include <png_utils.h>
#include "tasks.h"

tl::expected<std::pair<nf::render_config,
                       std::unique_ptr<nf::render_config_gpu_interface>>,
             std::string>
create_render_config_objects(const render_task& rt,
                             int archive_points) noexcept {
  nf::render_config render_config;
  {
    auto temp = nf::load_render_config_from_file(rt.render_config_filename);
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

tl::expected<void, std::string> render_gpu(const render_task& rt,
                                           const nf::newton_archive& ar,
                                           fu::unique_map& image) noexcept;

tl::expected<void, std::string> render_cpu(const render_task& rt,
                                           const nf::newton_archive& ar,
                                           fu::unique_map& image) noexcept;

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
                                             rt.archive_file.value().c_str(),
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

  {
    auto temp = check_archive(*src);
    if (!temp.has_value()) {
      return tl::make_unexpected(
          fmt::format("The archive is not valid! Detail: {}", temp.error()));
    }
  }

  fu::unique_map image_u8c3;
  tl::expected<void, std::string> err;
  if (rt.use_cpu) {
    err = render_cpu(rt, *src, image_u8c3);
  } else {
    err = render_gpu(rt, *src, image_u8c3);
  }
  if (!err.has_value()) {
    return tl::make_unexpected(err.error());
  }

  std::vector<const void*> row_ptrs;
  row_ptrs.reserve(src->info().rows);
  for (int r = rt.skip_rows; r < src->info().rows - rt.skip_rows; r++) {
    row_ptrs.emplace_back(image_u8c3.address<fu::pixel_RGB>(r, rt.skip_cols));
  }
  fmt::print("row_ptrs = [");
  for (auto ptr : row_ptrs) {
    fmt::print("{}, ", ptr);
  }
  fmt::print("]\n");

  if (!fu::write_png(rt.image_filename.c_str(), fu::color_space::u8c3,
                     row_ptrs.data(), src->info().rows - 2 * rt.skip_rows,
                     src->info().cols - 2 * rt.skip_cols)) {
    return tl::make_unexpected(fmt::format(
        "Function fu::write_png failed to generate \"{}\"", rt.image_filename));
  }
  return {};
}

tl::expected<void, std::string> render_gpu(
    const render_task& rt, const nf::newton_archive& ar,
    fu::unique_map& image_u8c3) noexcept {
  auto render_option_objects =
      create_render_config_objects(rt, ar.info().num_points());
  if (!render_option_objects.has_value()) {
    return tl::make_unexpected(
        fmt::format("Failed to make render config objects, detail: {}",
                    render_option_objects.error()));
  }

  if constexpr (false) {
    fmt::print("The render config is: {}\n",
               serialize_render_config(render_option_objects.value().first));
    fmt::print("The render config on gpu is: {}\n",
               serialize_render_config(
                   render_option_objects.value().second->config().value()));
  }

  std::unique_ptr<nf::gpu_render> renderer{nullptr};
  {
    auto temp = nf::gpu_render::create(ar.info().rows, ar.info().cols);

    if (!temp.has_value()) {
      return tl::make_unexpected(fmt::format(
          "Failed to create gpu_interface, detail: {}", temp.error()));
    }
    renderer = std::move(temp.value());
  }

  image_u8c3.reset((size_t)ar.info().rows, (size_t)ar.info().cols, 3);

  auto err = renderer->render(*render_option_objects.value().second,
                              ar.map_has_result(), ar.map_nearest_point_idx(),
                              ar.map_complex_difference(), image_u8c3,
                              rt.skip_rows, rt.skip_cols);
  if (!err.has_value()) {
    return tl::make_unexpected(
        fmt::format("Failed to render because ", err.error()));
  }

  return {};
}

tl::expected<void, std::string> render_cpu(
    const render_task& rt, const nf::newton_archive& ar,
    fu::unique_map& image_u8c3) noexcept {
  nf::render_config config;
  {
    auto render_config_opt =
        nf::load_render_config_from_file(rt.render_config_filename);
    if (!render_config_opt.has_value()) {
      return tl::make_unexpected(render_config_opt.error());
    }
    config = std::move(render_config_opt.value());
  }

  // fu::unique_map image_u8c3{(size_t)ar.info().rows, (size_t)ar.info().cols,
  // 3};
  image_u8c3.reset((size_t)ar.info().rows, (size_t)ar.info().cols, 3);

  nf::cpu_renderer renderer;
  renderer.render(config, ar.map_has_result(), ar.map_nearest_point_idx(),
                  ar.map_complex_difference(), image_u8c3, 0, 0);

  return {};
}