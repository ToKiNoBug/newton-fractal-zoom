#include <iostream>
#include <newton_archive.h>
#include <newton_fractal.h>
#include <fmt/format.h>
#include <newton_render.h>
#include <png_utils.h>
#include "run_compute.h"

tl::expected<std::pair<nf::render_config,
                       std::unique_ptr<nf::render_config_gpu_interface>>,
             std::string>
create_render_config_objects(const render_task& rt,
                             int archive_points) noexcept {
  nf::render_config render_config;
  {
    auto temp =
        nf::load_render_config(std::string_view{rt.render_config_filename});
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

tl::expected<void, std::string> render_gpu(
    const render_task& rt, const nf::newton_archive& ar) noexcept;

tl::expected<void, std::string> render_cpu(
    const render_task& rt, const nf::newton_archive& ar) noexcept;

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

  if (rt.use_cpu) {
    return render_cpu(rt, *src);
  } else {
    return render_gpu(rt, *src);
  }
}

tl::expected<void, std::string> render_gpu(
    const render_task& rt, const nf::newton_archive& ar) noexcept {
  auto render_option_objects =
      create_render_config_objects(rt, ar.info().num_points());
  if (!render_option_objects.has_value()) {
    return tl::make_unexpected(
        fmt::format("Failed to make render config objects, detail: {}",
                    render_option_objects.error()));
  }

  if constexpr (true) {
    fmt::print("The render config is: {}\n",
               serialize_render_config(render_option_objects.value().first));
    fmt::print("The render config on gpu is: {}\n",
               serialize_render_config(
                   render_option_objects.value().second->config().value()));
  }

  std::unique_ptr<nf::gpu_interface> gi{nullptr};
  {
    auto temp = nf::gpu_interface::create(ar.info().rows, ar.info().cols);

    if (!temp.has_value()) {
      return tl::make_unexpected(fmt::format(
          "Failed to create gpu_interface, detail: {}", temp.error()));
    }
    gi = std::move(temp.value());
  }

  fu::unique_map image_u8c3{(size_t)ar.info().rows, (size_t)ar.info().cols, 3};

#define NF_NFTOOL_HANDEL_ERROR(err) \
  if (!err.has_value()) return tl::make_unexpected(err.error());

  {
    auto err = gi->set_has_value(ar.map_has_result());
    NF_NFTOOL_HANDEL_ERROR(err);
    err = gi->set_nearest_index(ar.map_nearest_point_idx());
    NF_NFTOOL_HANDEL_ERROR(err);
    err = gi->set_complex_difference(ar.map_complex_difference());
    NF_NFTOOL_HANDEL_ERROR(err);

    gi->wait_for_finished();

    err = gi->run(*render_option_objects.value().second, 0, 0, false);
    NF_NFTOOL_HANDEL_ERROR(err);

    err = gi->get_pixels(image_u8c3);
    NF_NFTOOL_HANDEL_ERROR(err);
  }

  if (!fu::write_png(rt.image_filename.c_str(), fu::color_space::u8c3,
                     image_u8c3)) {
    return tl::make_unexpected(fmt::format(
        "Function fu::write_png failed to generate \"{}\"", rt.image_filename));
  }
  return {};
}

tl::expected<void, std::string> render_cpu(
    const render_task& rt, const nf::newton_archive& ar) noexcept {
  nf::render_config config;
  {
    auto render_config_opt =
        nf::load_render_config(std::string_view{rt.render_config_filename});
    if (!render_config_opt.has_value()) {
      return tl::make_unexpected(render_config_opt.error());
    }
    config = std::move(render_config_opt.value());
  }

  fu::unique_map image_u8c3{(size_t)ar.info().rows, (size_t)ar.info().cols, 3};

  nf::cpu_renderer renderer;
  renderer.render(config, ar.map_has_result(), ar.map_nearest_point_idx(),
                  ar.map_complex_difference(), image_u8c3, 0, 0);

  if (!fu::write_png(rt.image_filename.c_str(), fu::color_space::u8c3,
                     image_u8c3)) {
    return tl::make_unexpected(fmt::format(
        "Function fu::write_png failed to generate \"{}\"", rt.image_filename));
  }
  return {};
}