//
// Created by joseph on 6/20/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_RUN_COMPUTE_H
#define NEWTON_FRACTAL_ZOOM_RUN_COMPUTE_H

#include <string>
#include <optional>
#include <tl/expected.hpp>
#include <thread>

struct compute_task {
  compute_task() : threads{std::thread::hardware_concurrency()} {}

  std::string filename;
  std::optional<int> row_override{std::nullopt};
  std::optional<int> col_override{std::nullopt};
  std::optional<int> iteration_override{std::nullopt};

  std::optional<int> precision_override{std::nullopt};

  std::string archive_filename;
  uint32_t threads;
};

tl::expected<void, std::string> run_compute(const compute_task& ct) noexcept;

#endif  // NEWTON_FRACTAL_ZOOM_RUN_COMPUTE_H
