#include <newton_fractal.h>
#include <CLI11.hpp>
#include <fstream>
#include <fmt/format.h>

int main(int argc, char** argv) {
  CLI::App app;
  std::string filename;
  app.add_option("task", filename)->required()->check(CLI::ExistingFile);
  CLI11_PARSE(app, argc, argv);
  std::ifstream ifs{filename};
  auto mie = newton_fractal::load_metadata(ifs, false);

  if (!mie.has_value()) {
    fmt::print("Failed to load meta info: {}", mie.error());
  }

  auto& mi = mie.value();

  for (int idx = 0; idx < mi.num_points(); idx++) {
    auto temp = mi.equation()->point_at(idx);
    fmt::print("Point at index {} : [{}, {}]\n", idx, temp.real(), temp.imag());
  }

  return 0;
}