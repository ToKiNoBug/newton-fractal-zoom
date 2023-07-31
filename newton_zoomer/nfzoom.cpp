
#include "newton_label.h"
#include <CLI11.hpp>
#include <QApplication>
#include <thread>
#include <omp.h>
#include "newton_zoomer.h"
#include <filesystem>

namespace stdfs = std::filesystem;

bool can_be_file(stdfs::path path) noexcept {
  return stdfs::is_regular_file(path) || stdfs::is_symlink(path);
}

nf::cpu_renderer cpu_renderer;
tl::expected<std::unique_ptr<nf::gpu_render>, std::string> gpu_renderer_exp =
    nf::gpu_render::create(100, 100);
tl::expected<std::unique_ptr<nf::render_config_gpu_interface>, std::string>
    gpu_render_config_exp = nf::render_config_gpu_interface::create();

int main(int argc, char **argv) {
  QApplication qapp{argc, argv};

  CLI::App capp;
  capp.set_version_flag("--version", NEWTON_FRACTAL_VERSION_STR);

  std::string compute_src{"../compute_presets/double-p3.json"};
  std::string render_config{"../render_presets/plasma-10.json"};
  int scale{1};
  uint32_t threads{std::thread::hardware_concurrency()};
  bool auto_precision{false};
  bool render_cuda{false};
  capp.add_option("source", compute_src, "The staring point of zooming")
      ->check(CLI::ExistingFile);
  capp.add_option("--rj,--render-json", render_config,
                  "The method of rendering")
      ->check(CLI::ExistingFile);
  capp.add_option("-j,--threads", threads, "Threads used to compute and render")
      ->default_val(std::thread::hardware_concurrency());
  capp.add_option("--scale", scale,
                  "The scaling ratio that the image will be displayed")
      ->default_val(1)
      ->check(CLI::PositiveNumber);
  capp.add_flag("--render-cuda", render_cuda, "Render with cuda")
      ->default_val(false);
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
  capp.add_flag("--auto-precision", auto_precision,
                "Change the precision automatically. Only available for mpfr.")
      ->default_val(false);
#endif
  CLI11_PARSE(capp, argc, argv);

  if (!can_be_file(compute_src)) {
    fmt::print("{} is not a file.", compute_src);
    return 1;
  }
  if (!can_be_file(render_config)) {
    fmt::print("{} is not a file.", render_config);
    return 1;
  }
  omp_set_num_threads(int(threads));

  newton_zoomer zoomer;
  zoomer.set_scale(scale);
  zoomer.auto_precision = auto_precision;
  zoomer.gpu_render = render_cuda;

  if (render_cuda) {
    if (!::gpu_renderer_exp) {
      fmt::print("Failed to create cuda renderer because {}\n",
                 gpu_renderer_exp.error());
      return 2;
    }
    if (!::gpu_render_config_exp) {
      fmt::print("Failed to create cuda render config object because {}\n",
                 gpu_render_config_exp.error());
      return 2;
    }
  }

  {
    auto mi = nf::load_metadata_from_file(compute_src, false);
    if (!mi) {
      fmt::print("Failed to load metadata from file \"{}\", detail: {}",
                 compute_src, mi.error());
      return 1;
    }
    zoomer.set_template_metadata(std::move(mi.value()));
  }
  {
    auto rj = nf::load_render_config_from_file(render_config);
    if (!rj) {
      fmt::print("Failed to load render config from file \"{}\", detail: {}",
                 render_config, rj.error());
      return 2;
    }
    zoomer.render_config = std::move(rj.value());
  }

  // zoomer.template_metadata().window()->copy_to(zoomer.current_result().wind.get());
  zoomer.refresh_range_display();
  // zoomer.label_widget()->

  zoomer.show();

  {
    QLocale locale;
    if (locale.language() == QLocale::Language::Chinese) {
      zoomer.set_language(fu::language_t::zh_CN);
    }
  }

  return qapp.exec();
}
