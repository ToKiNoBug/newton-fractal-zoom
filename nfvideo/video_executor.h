#ifndef NEWTON_FRACTAL_ZOOM_VIDEO_EXECUTOR_H
#define NEWTON_FRACTAL_ZOOM_VIDEO_EXECUTOR_H

#include <video_utils.h>
#include "load_video_task.h"

class video_executor : public fu::video_executor_base {
 protected:
  [[nodiscard]] std::unique_ptr<fu::common_info_base> load_common_info(
      std::string &err) const noexcept final;
  [[nodiscard]] std::unique_ptr<fu::compute_task_base> load_compute_task(
      std::string &err) const noexcept final;
  [[nodiscard]] std::unique_ptr<fu::render_task_base> load_render_task(
      std::string &err) const noexcept final;
  [[nodiscard]] std::unique_ptr<fu::video_task_base> load_video_task(
      std::string &err) const noexcept final;

 public:
  [[nodiscard]] std::optional<fu::full_task> load_task(
      std::string &err) const noexcept final;

  void compute(int archive_idx, const fu::wind_base &window,
               std::any &ret) const noexcept final;

  [[nodiscard]] std::unique_ptr<fu::render_resource_base>
  create_render_resource() const noexcept final;

  [[nodiscard]] std::string render_with_skip(
      const std::any &archive, int archive_index, int image_idx, int skip_rows,
      int skip_cols, fu::map_view image_u8c3,
      fu::render_resource_base *resource) const noexcept final;

  [[nodiscard]] err_info_t save_archive(
      const std::any &, std::string_view filename) const noexcept final;

  [[nodiscard]] err_info_t error_of_archive(
      std::string_view filename, std::any &archive) const noexcept final;

  [[nodiscard]] std::string load_archive(
      std::string_view filename, std::span<uint8_t> buffer,
      std::any &archive) const noexcept final;

 public:
  std::string task_file{""};
  bool load_archive_as_render_mode{true};
  bool use_gpu{true};
};

#endif  // NEWTON_FRACTAL_ZOOM_VIDEO_EXECUTOR_H
