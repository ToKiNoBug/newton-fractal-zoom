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

  auto look = capp.add_subcommand("look");
  look_task lt;
  {
    look->add_option("archive", lt.source_file)
        ->required()
        ->check(CLI::ExistingFile);
    look->add_flag("--load-as-render-mode,--lrm", lt.load_as_render_mode);
    look->add_flag("--show-metainfo,--smi", lt.show_metainfo);
    look->add_option("--extract-metainfo,--emi", lt.extract_metainfo);
    look->add_option("--extract-has-value,--ehv", lt.extract_has_value);
    look->add_option("--extract-nearest-index,--eni", lt.extract_nearest_index);
    look->add_option("--extract-complex-diff,--ecd",
                     lt.extract_complex_difference);
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

  if (look->count() > 0) {
    auto result = run_look(lt);
    if (!result.has_value()) {
      fmt::print("Failed to lookup. Detail: {}\n", result.error());
      return 1;
    }
  }

  return 0;
}