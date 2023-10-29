//
// Created by David on 2023/7/8.
//

#ifndef NEWTON_FRACTAL_ZOOM_LOAD_VIDEO_TASK_H
#define NEWTON_FRACTAL_ZOOM_LOAD_VIDEO_TASK_H

#include <newton_archive.h>
#include <newton_render.h>
#include <video_utils.h>
#include <tl/expected.hpp>
#include <set>

class video_executor;

class common_info : public fractal_utils::common_info_base {
 public:
  nf::meta_data metadata;

  [[nodiscard]] size_t rows() const noexcept final {
    return this->metadata.rows;
  }
  [[nodiscard]] size_t cols() const noexcept final {
    return this->metadata.cols;
  }
};

tl::expected<common_info, std::string> load_common_info();

class compute_task : public fractal_utils::compute_task_base {
 private:
  common_info *related_ci{nullptr};
  friend class video_executor;

 public:
  std::set<int> no_check_frames;

  [[nodiscard]] fu::wind_base *start_window() noexcept final {
    return this->related_ci->metadata.window();
  }
  [[nodiscard]] const fu::wind_base *start_window() const noexcept final {
    return this->related_ci->metadata.window();
  }
  [[nodiscard]] bool need_check_frame(int frameid) const noexcept {
    return !this->no_check_frames.contains(frameid);
  }
  //  std::variant<thin_compute_objs, nf::meta_data::non_compute_info>
  //      thin_metainfo;
};

class render_task : public fu::render_task_base {
 public:
  nf::render_config render_config;
};

class video_task : public fu::video_task_base {};

class render_resource : public fu::render_resource_base {
 public:
  struct gpu_render_suit {
    std::unique_ptr<nf::gpu_render> renderer{nullptr};
    std::unique_ptr<nf::render_config_gpu_interface> config{nullptr};
  };

  std::variant<nf::cpu_renderer, gpu_render_suit> renderer;
};

#endif  // NEWTON_FRACTAL_ZOOM_LOAD_VIDEO_TASK_H
