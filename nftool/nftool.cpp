#include <nlohmann/json.hpp>
#include <CLI11.hpp>
#include <newton_fractal.h>
#include <optional>
#include <fmt/format.h>
#include "run_compute.h"
#include <newton_archive.h>
#include <string>

int main(int argc, char** argv) {
  CLI::App capp;

  auto compute = capp.add_subcommand("compute");

  compute_task ct;
  {
    compute->add_option("meta-info-file", ct.filename, "Task to compute")
        ->check(CLI::ExistingFile)
        ->required();
    compute->add_option("-o", ct.archive_filename);
    compute->add_option("--rows", ct.row_override);
    compute->add_option("-j,--threads", ct.threads);
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
    compute->add_flag("--track-memory", ct.track_memory)->default_val(false);
#endif
  }

  auto render = capp.add_subcommand("render");
  render_task rt;
  {
    render->add_option("archive", rt.archive_file, "The archive to render")
        ->check(CLI::ExistingFile);
    render->add_option("-o", rt.image_filename);
    render->add_option("--rj,--render-json", rt.render_config_filename);
  }

  CLI11_PARSE(capp, argc, argv);

  nf::newton_archive archive;

  if (compute->count() > 0) {
    if (render->count() > 0) {
      ct.return_archive = &archive;
      rt.archive_value = &archive;
    }
    auto result = run_compute(ct);
    if (!result.has_value()) {
      fmt::print("Computation failed. Detail: {}\n", result.error());
      return 1;
    }
  }

  if (render->count() > 0) {
    auto result = run_render(rt);
    if (!result.has_value()) {
      fmt::print("Render failed. Detail: {}\n", result.error());
      return 1;
    }
  }

  return 0;
}