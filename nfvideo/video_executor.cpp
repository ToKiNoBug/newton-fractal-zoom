#include "video_executor.h"
#include "load_video_task.h"
#include <newton_archive.h>
#include <newton_render.h>
#include <omp.h>

void video_executor::compute(int archive_idx, const fu::wind_base &window,
                             std::any &ret) const noexcept {
  if (std::any_cast<nf::newton_archive>(&ret) == nullptr) {
    ret.emplace<nf::newton_archive>(nf::newton_archive{});
  }

  auto &ar = std::any_cast<nf::newton_archive &>(ret);

  ar.info() = dynamic_cast<common_info *>(this->task().common.get())->metadata;
  auto copy_success = window.copy_to(ar.info().window());
  if (!copy_success) {
    fmt::print("window.copy_to failed.\n");
    exit(1);
    return;
  }
  if (!ar.info().obj_creator()->is_fixed_precision()) {
    ar.info().set_precision(ar.info().obj_creator()->suggested_precision_of(
        window, ar.info().rows, ar.info().cols));
  }

  ar.setup_matrix();

  //  fmt::print("center = [{}, {}]\n",
  //  ar.info().window()->displayed_center()[0],
  //             ar.info().window()->displayed_center()[1]);
  // omp_set_num_threads(this->task().compute->threads);
  nf::newton_equation_base::compute_option opt{
      .bool_has_result = ar.map_has_result(),
      .u8_nearest_point_idx = ar.map_nearest_point_idx(),
      .f64complex_difference = ar.map_complex_difference()};
  ar.info().equation()->compute(*ar.info().window(), ar.info().iteration, opt);
}

std::unique_ptr<fu::render_resource_base>
video_executor::create_render_resource() const noexcept {
  render_resource temp;

  if (!this->use_gpu) {
    temp.renderer.emplace<nf::cpu_renderer>(nf::cpu_renderer{});
    return std::make_unique<render_resource>(std::move(temp));
  }

  render_resource::gpu_render_suit suit;
  {
    auto renderer = nf::gpu_render::create(this->task().common->rows(),
                                           this->task().common->cols());
    if (!renderer) {
      fmt::print("Failed to create gpu_renderer because {}\n",
                 renderer.error());
      return nullptr;
    }
    suit.renderer = std::move(renderer.value());
  }
  {
    auto config = nf::render_config_gpu_interface::create();
    if (!config) {
      fmt::print(
          "Failed to create nf::render_config_gpu_interface because {}\n",
          config.error());
      return nullptr;
    }
    suit.config = std::move(config.value());
  }
  temp.renderer.emplace<render_resource::gpu_render_suit>(std::move(suit));

  return std::make_unique<render_resource>(std::move(temp));
}

std::string video_executor::render_with_skip(
    const std::any &archive, int archive_index, int image_idx, int skip_rows,
    int skip_cols, fu::map_view image_u8c3,
    fu::render_resource_base *resource_base) const noexcept {
  const auto *arp = std::any_cast<nf::newton_archive>(&archive);
  if (arp == nullptr) {
    return "The given const std::any & doesn't contain an archive object. ";
  }
  auto *resource = dynamic_cast<render_resource *>(resource_base);
  if (resource == nullptr) {
    return "The render resource ptr is null, or not an instance of "
           "::render_resource.";
  }

  thread_local int prev_archive_index = -1;
  bool need_set_data = false;
  if (prev_archive_index != archive_index) {
    need_set_data = true;
    prev_archive_index = archive_index;
  }
  const auto &config =
      dynamic_cast<render_task *>(this->task().render.get())->render_config;
  if (resource->renderer.index() == 0) {
    auto &cpu_renderer = std::get<0>(resource->renderer);
    if (need_set_data) {
      cpu_renderer.set_data(arp->map_has_result(), arp->map_nearest_point_idx(),
                            arp->map_complex_difference(), false);
    }
    auto err = cpu_renderer.render(config, image_u8c3, skip_rows, skip_cols);
    if (!err) {
      return err.error();
    }
  } else {
    auto &suit = std::get<1>(resource->renderer);
    thread_local bool is_config_passed_to_gpu = false;
    if (!is_config_passed_to_gpu) {
      is_config_passed_to_gpu = true;
      auto err = suit.config->set_config(config);
      if (!err) {
        return fmt::format("Failed to pass render config to gpu because {}",
                           err.error());
      }
    }
    if (need_set_data) {
      auto err = suit.renderer->set_data(arp->map_has_result(),
                                         arp->map_nearest_point_idx(),
                                         arp->map_complex_difference(), false);
      if (!err) {
        return fmt::format("gpu_renderer->set_data because {}", err.error());
      }
    }
    auto err =
        suit.renderer->render(*suit.config, image_u8c3, skip_rows, skip_cols);
    if (!err) {
      return fmt::format("gpu_renderer->set_data because {}", err.error());
    }
  }

  return {};
}

std::string video_executor::save_archive(
    const std::any &ar_, std::string_view filename) const noexcept {
  const auto *arp = std::any_cast<nf::newton_archive>(&ar_);
  if (arp == nullptr) {
    return "The given const std::any & doesn't contain a newton archive.";
  }

  auto err = arp->save(filename);
  if (!err) {
    return fmt::format("Failed to save archive because {}", err.error());
  }
  return {};
}

std::string video_executor::error_of_archive(std::string_view filename,
                                             std::any &ar_) const noexcept {
  const auto *arp = std::any_cast<nf::newton_archive>(&ar_);
  if (arp == nullptr) {
    return "The given const std::any & doesn't contain a newton archive.";
  }
  auto err = nf::check_archive(*arp);
  if (!err) {
    return err.error();
  }
  return {};
}

std::string video_executor::load_archive(std::string_view filename,
                                         std::span<uint8_t> buffer,
                                         std::any &archive) const noexcept {
  if (std::any_cast<nf::newton_archive>(&archive) == nullptr) {
    archive.emplace<nf::newton_archive>(nf::newton_archive{});
  }

  auto &ar = std::any_cast<nf::newton_archive &>(archive);
  nf::newton_archive::load_options load_opt;
  auto err =
      ar.load(filename, this->load_archive_as_render_mode, buffer, load_opt);

  if (!err) {
    archive.reset();
    return err.error();
  }
  return {};
}