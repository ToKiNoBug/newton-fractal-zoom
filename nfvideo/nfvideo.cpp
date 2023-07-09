#include <CLI11.hpp>
#include "load_video_task.h"
#include "video_executor.h"
#include <fmt/format.h>

int main(int argc, char** argv) {
  video_executor ve;

  CLI::App app;
  ve.task_file = "nfvideo.toml";
  app.add_option("task file", ve.task_file)
      ->check(CLI::ExistingFile)
      ->check(CLI::Validator{[](std::string& in) -> std::string {
                               if (in.ends_with(".toml")) return {};
                               return "Extension must be .toml";
                             },
                             "Extension must be .toml", "Extension check"});

  auto compute = app.add_subcommand("compute");
  auto render = app.add_subcommand("render");
  render->add_flag("--gpu", ve.use_gpu, "Render with gpu.")->default_val(true);

  auto mkvideo = app.add_subcommand("mkvideo");
  bool dry_run{false};
  mkvideo
      ->add_flag("--dry-run", dry_run,
                 "Display command instead of executing them.")
      ->default_val(false);

  CLI11_PARSE(app, argc, argv);

  ve.load_archive_as_render_mode = true;
  if (compute->count() > 0) {
    ve.load_archive_as_render_mode = false;
  }

  {
    std::string err;
    auto temp = ve.load_task(err);
    if (!temp.has_value()) {
      fmt::print(
          "Failed to load task, the task file may be invalid.\nDetail: {}\n",
          err);
      return 1;
    }
    ve.set_task(std::move(temp.value()));
  }

  if (compute->count() > 0) {
    const auto success = ve.run_compute();
    if (!success) {
      fmt::print("Computation failed.\n");
      return 1;
    }
  }

  if (render->count() > 0) {
    const auto success = ve.run_render();
    if (!success) {
      fmt::print("Render failed.\n");
      return 1;
    }
  }

  if (mkvideo->count() > 0) {
    const auto success = ve.make_video(dry_run);
    if (!success) {
      fmt::print("Make video failed.\n");
      return 1;
    }
  }

  return 0;
}