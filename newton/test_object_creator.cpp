//
// Created by joseph on 6/19/23.
//

#include "newton_fractal.h"
#include "object_creator.h"
#include <magic_enum/magic_enum.hpp>
#include <fmt/format.h>

int main(int argc, char** argv) {
  for (auto lib : magic_enum::enum_values<fu::float_backend_lib>()) {
    for (auto prec : {1, 2, 4, 8, 16, 200}) {
      auto ocp = nf::object_creator::create(lib, prec, nf::gpu_backend::no);

      fmt::print("Testing lib = {}, precision = {}", magic_enum::enum_name(lib),
                 prec);

      if (ocp.has_value()) {
        fmt::print(" --success\n");
      } else {
        fmt::print(" --failed, detail: {}\n", ocp.error());
        continue;
      }
    }
  }

  return 0;
}