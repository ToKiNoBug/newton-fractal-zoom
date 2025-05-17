#include "tasks.h"
#include <magic_enum/magic_enum.hpp>
#include <newton_fractal.h>
#include <fmt/format.h>
#include <fstream>

tl::expected<void, std::string> run_task_cvt(
    const task_cvt_task& tct) noexcept {
  const auto fsf_opt =
      magic_enum::enum_cast<nf::float_save_format>(tct.object_format);
  if (!fsf_opt.has_value()) {
    return tl::make_unexpected(fmt::format(
        "Invalid format: \"{}\" is not a valid format.", tct.object_format));
  }

  const auto fsf = fsf_opt.value();

  auto metadata_exp = nf::load_metadata_from_file(tct.input_task, false);
  if (!metadata_exp.has_value()) {
    return tl::make_unexpected(fmt::format("Failed to load \"{}\" because ",
                                           tct.input_task,
                                           metadata_exp.error()));
  }

  auto& metadata = metadata_exp.value();

  auto err = save_metadata(metadata, fsf);
  if (!err) {
    return tl::make_unexpected(fmt::format("Failed to save \"{}\" because {}",
                                           tct.out_file, err.error()));
  }

  std::ofstream ofs{tct.out_file};
  if (!ofs) {
    return tl::make_unexpected(
        fmt::format("Failed to create or open \"{}\"", tct.out_file));
  }
  ofs << err.value().dump(2);
  ofs.close();

  return {};
}