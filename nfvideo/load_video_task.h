//
// Created by David on 2023/7/8.
//

#ifndef NEWTON_FRACTAL_ZOOM_LOAD_VIDEO_TASK_H
#define NEWTON_FRACTAL_ZOOM_LOAD_VIDEO_TASK_H

#include <newton_archive.h>
#include <newton_render.h>
#include <video_utils.h>
#include <tl/expected.hpp>

class common_info : public fractal_utils::common_info_base {
 public:
};

tl::expected<common_info, std::string> load_common_info();

struct thin_compute_objs {
  std::unique_ptr<nf::object_creator> obj_creator{nullptr};
  std::unique_ptr<nf::newton_equation_base> equation{nullptr};
};

class compute_task : public fractal_utils::compute_task_base {
 public:
  std::variant<thin_compute_objs, nf::meta_data::non_compute_info>
      thin_metainfo;
};

class render_task : public fu::render_task_base {
 public:
  nf::render_config render_config;
};

class video_task : public fu::video_task_base {};

class render_resource : public fu::render_task_base {
 public:
  struct gpu_render_suit {
    std::unique_ptr<nf::gpu_render> renderer{nullptr};
    std::unique_ptr<nf::render_config_gpu_interface> config{nullptr};
  };

  std::variant<nf::cpu_renderer, gpu_render_suit> renderer;
};

#endif  // NEWTON_FRACTAL_ZOOM_LOAD_VIDEO_TASK_H
