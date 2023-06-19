
#include "run_compute.h"
#include <newton_fractal.h>
#include <fstream>
#include <fmt/format.h>

tl::expected<void, std::string> run_compute(const compute_task& ct) noexcept {
  nf::meta_data metadata;
  {
    std::ifstream ifs{ct.filename};
    auto md_e = nf::load_metadata(ifs);
    if (!md_e.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to laod meta data. Detail: {}", md_e.error()));
    }
    metadata = std::move(md_e.value());
  }

  return {};
}