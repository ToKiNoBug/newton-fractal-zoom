#include <CLI11.hpp>
#include <optional>
#include <fmt/format.h>
#include <newton_archive.h>
#include <string>
#include <nlohmann/json.hpp>
#include <newton_fractal.h>
#include <magic_enum.hpp>

#include "tasks.h"
int main(int argc, char** argv) {
  CLI::App capp;
  capp.set_version_flag("--version", NEWTON_FRACTAL_VERSION_STR);

  auto compute = capp.add_subcommand("compute");

  compute_task ct;
  {
    compute->add_option("meta-info-file", ct.filename, "Task to compute")
        ->check(CLI::ExistingFile)
        ->required();
    compute->add_option("-o", ct.archive_filename,
                        "The filename of generated archive");
    compute
        ->add_option("-j,--threads", ct.threads,
                     "Number of threads for computation.")
        ->default_val(std::thread::hardware_concurrency());
    compute->add_option("--rows", ct.row_override,
                        "Rows that overrides the default value.");
    compute
        ->add_option("--cols", ct.col_override,
                     "Cols that overrides the default value.")
        ->default_val(std::nullopt);
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
    compute->add_option(
        "-p,--precision", ct.precision_override,
        "Precision that override the default value. Only available for mpfr.");
    compute
        ->add_flag("--track-memory", ct.track_memory,
                   "Record the calling of malloc and realloc by gmp, only "
                   "available for mpfr.")
        ->default_val(false);
#endif
  }

  auto render = capp.add_subcommand("render");
  render_task rt;
  {
    render->add_option("archive", rt.archive_file, "The archive to render")
        ->check(CLI::ExistingFile);
    render->add_option("-o", rt.image_filename);
    render->add_option("--rj,--render-json", rt.render_config_filename);
    render->add_flag("--cpu", rt.use_cpu)->default_val(false);
    render->add_option("--skip-rows", rt.skip_rows)
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
    render->add_option("--skip-cols", rt.skip_cols)
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
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

  auto taskcvt = capp.add_subcommand("taskcvt");
  task_cvt_task tct;
  {
    taskcvt->add_option("input_file", tct.input_task, "Input task file")
        ->required()
        ->check(CLI::ExistingFile);
    taskcvt->add_option("--format,-f", tct.object_format, "Object format")
        ->required()
        ->check(CLI::IsMember(magic_enum::enum_names<nf::float_save_format>()));
    taskcvt->add_option("-o", tct.out_file, "Generated file.")->required();
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

  if (taskcvt->count() > 0) {
    auto result = run_task_cvt(tct);
    if (!result) {
      fmt::print("Failed to convert task file. Detail: {}\n", result.error());
      return 1;
    }
  }

  return 0;
}