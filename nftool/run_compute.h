//
// Created by joseph on 6/20/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_RUN_COMPUTE_H
#define NEWTON_FRACTAL_ZOOM_RUN_COMPUTE_H

#include <string>
#include <optional>
#include <tl/expected.hpp>
#include <thread>

namespace newton_fractal {
class newton_archive;
}
struct compute_task {
  compute_task() : threads{std::thread::hardware_concurrency()} {}

  std::string filename;
  std::optional<int> row_override{std::nullopt};
  std::optional<int> col_override{std::nullopt};
  std::optional<int> iteration_override{std::nullopt};

  std::optional<int> precision_override{std::nullopt};

  std::string archive_filename;
  uint32_t threads;
  bool track_memory{false};
  newton_fractal::newton_archive* return_archive{nullptr};
};
tl::expected<void, std::string> run_compute(const compute_task& ct) noexcept;

struct render_task {
  std::optional<std::string> archive_file{std::nullopt};
  newton_fractal::newton_archive* archive_value{nullptr};
  std::string render_config_filename;

  std::string image_filename;
};
tl::expected<void, std::string> run_render(const render_task& rt) noexcept;

struct look_task {
  std::string source_file;
  bool load_as_render_mode{false};
  bool show_metainfo{false};
  std::string extract_metainfo;
  std::string extract_has_value;
  std::string extract_nearest_index;
  std::string extract_complex_difference;
};
tl::expected<void, std::string> run_look(const look_task& lt) noexcept;

#endif  // NEWTON_FRACTAL_ZOOM_RUN_COMPUTE_H
