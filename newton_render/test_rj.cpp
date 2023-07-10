#include <CLI11.hpp>
#include <newton_render.h>
#include <fmt/format.h>

int main(int argc, char** argv) {
  CLI::App app;
  std::string rj_filename;
  app.add_option("render_json", rj_filename)
      ->required()
      ->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  auto rc_exp = newton_fractal::load_render_config_from_file(rj_filename);
  if (!rc_exp) {
    fmt::print("{}\n", rc_exp.error());
    return 1;
  }

  return 0;
}