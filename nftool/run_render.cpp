#include "run_compute.h"
#include <newton_archive.h>
#include <newton_fractal.h>
#include <fmt/format.h>

tl::expected<void, std::string> run_render(const render_task& rt) noexcept {
  std::optional<nf::newton_archive> archive{std::nullopt};
  const nf::newton_archive* src{nullptr};
  if (!rt.archive_file.has_value()) {
    if (rt.archive_value != nullptr) {
      src = rt.archive_value;
    }
  } else {
    auto temp =
        nf::newton_archive::load_archive(rt.archive_file.value(), true, {});

    if (!temp.has_value()) {
      return tl::make_unexpected(fmt::format("Failed to load {}, detail: {}",
                                             rt.archive_file.value(),
                                             temp.error()));
    }
    archive = std::move(temp.value());
    src = &archive.value();
  }
  if (src == nullptr) {
    return tl::make_unexpected(
        "No value for source. Please assign the source filename, or run a "
        "computation together.");
  }

  return {};
}