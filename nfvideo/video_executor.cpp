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

  ar.info().set_precision(ar.info().obj_creator()->suggested_precision_of(
      window, ar.info().rows, ar.info().cols));

  ar.setup_matrix();
  omp_set_num_threads(this->task().compute->threads);
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

std::string video_executor::render(
    const std::any &archive, int archive_index, int image_idx,
    fu::map_view image_u8c3,
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