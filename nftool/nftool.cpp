#include <nlohmann/json.hpp>
#include <CLI11.hpp>
#include <newton_fractal.h>
#include <optional>
#include <fmt/format.h>
#include "run_compute.h"

int main(int argc, char** argv) {
  CLI::App capp;

  auto compute = capp.add_subcommand("compute");

  compute_task ct;
  {
    compute->add_option("meta-info-file", ct.filename, "Task to compute")
        ->check(CLI::ExistingFile)
        ->required();
    compute->add_option("-o", ct.archive_filename)->required();
    compute->add_option("--rows", ct.row_override);
  }

  CLI11_PARSE(capp, argc, argv);

  if (compute->count() > 0) {
    auto result = run_compute(ct);
    if (!result.has_value()) {
      fmt::print("Computation failed. Detail: {}\n", result.error());
      return 1;
    }
  }

  return 0;
}