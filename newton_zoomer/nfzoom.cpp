
#include "newton_label.h"
#include <CLI11.hpp>
#include <QApplication>
#include <thread>
#include <omp.h>
#include "newton_zoomer.h"

int main(int argc, char** argv) {
  QApplication qapp{argc, argv};

  CLI::App capp;
  std::string compute_src;
  std::string render_config;
  int scale{1};
  uint32_t threads{std::thread::hardware_concurrency()};
  bool auto_precision{false};
  capp.add_option("source", compute_src)->check(CLI::ExistingFile)->required();
  capp.add_option("--rj,--render-json", render_config)
      ->check(CLI::ExistingFile)
      ->required();
  capp.add_option("-j,--threads", threads)
      ->default_val(std::thread::hardware_concurrency());
  capp.add_option("--scale", scale)->default_val(1);
  capp.add_flag("--auto-precision", auto_precision)->default_val(false);

  CLI11_PARSE(capp, argc, argv);

  omp_set_num_threads(int(threads));

  newton_zoomer zoomer;
  zoomer.set_scale(scale);
  zoomer.auto_precision = auto_precision;

  {
    auto mi = nf::load_metadata_from_file(compute_src, false);
    if (!mi) {
      fmt::print("Failed to load metadata, detail: {}", mi.error());
      return 1;
    }
    zoomer.set_template_metadata(std::move(mi.value()));
  }
  {
    auto rj = nf::load_render_config_from_file(render_config);
    if (!rj) {
      fmt::print("Failed to load render config, detail: {}", rj.error());
      return 2;
    }
    zoomer.render_config = std::move(rj.value());
  }

  zoomer.show();

  return qapp.exec();
}