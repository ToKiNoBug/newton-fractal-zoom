#include "tasks.h"
#include <newton_archive.h>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <fstream>

void print_meta_info(const nf::meta_data&) noexcept;

tl::expected<void, std::string> save_data(std::string_view filename,
                                          fu::constant_view) noexcept;
tl::expected<void, std::string> save_data(std::string_view filename,
                                          std::span<const uint8_t>) noexcept;

tl::expected<void, std::string> run_look(const look_task& lt) noexcept {
  nf::newton_archive nar;
  fu::binary_archive bar;
  {
    auto temp =
        nar.load(lt.source_file, lt.load_as_render_mode, {},
                 nf::newton_archive::load_options{.return_archive = &bar});

    if (!temp.has_value()) {
      return tl::make_unexpected(fmt::format("Failed to load {} because {}",
                                             lt.source_file, temp.error()));
    }
  }

  if (lt.show_metainfo) {
    print_meta_info(nar.info());
  }

  if (!lt.extract_metainfo.empty()) {
    auto seg =
        bar.find_first_of((int64_t)nf::newton_archive::data_tag::metadata);
    assert(seg != nullptr);

    auto err = save_data(lt.extract_metainfo,
                         {(const uint8_t*)seg->data(), seg->bytes()});
    if (!err.has_value()) {
      return tl::make_unexpected(err.error());
    }
  }

  if (!lt.extract_has_value.empty()) {
    auto err = save_data(lt.extract_has_value, nar.map_has_result());
    if (!err.has_value()) {
      return tl::make_unexpected(err.error());
    }
  }

  if (!lt.extract_nearest_index.empty()) {
    auto err = save_data(lt.extract_nearest_index, nar.map_nearest_point_idx());
    if (!err.has_value()) {
      return tl::make_unexpected(err.error());
    }
  }

  if (!lt.extract_complex_difference.empty()) {
    auto err =
        save_data(lt.extract_complex_difference, nar.map_complex_difference());
    if (!err.has_value()) {
      return tl::make_unexpected(err.error());
    }
  }

  return {};
}

void print_meta_info(const nf::meta_data& mi) noexcept {
  fmt::print("rows = {}, cols = {}, iteration = {}\n", mi.rows, mi.cols,
             mi.iteration);

  fmt::print("num_points = {}\n", mi.num_points());
  if (mi.compute_objs.index() == 1) {
    return;
  }

  const auto& co = std::get<nf::meta_data::compute_objects>(mi.compute_objs);

  fmt::print("floating point type = {}, precision = {}\n",
             magic_enum::enum_name(co.obj_creator->backend_lib()),
             co.obj_creator->precision());

  fmt::print("Equation: {}\n", co.equation->to_string());
}

tl::expected<void, std::string> save_data(
    std::string_view filename, std::span<const uint8_t> data) noexcept {
  std::ofstream ofs{filename.data(), std::ios::binary};
  if (!ofs) {
    return tl::make_unexpected(
        fmt::format("Failed to create or open {}", filename));
  }

  ofs.write((const char*)data.data(), data.size_bytes());
  ofs.close();
  return {};
}

tl::expected<void, std::string> save_data(std::string_view filename,
                                          fu::constant_view mat) noexcept {
  return save_data(filename, {(const uint8_t*)mat.data(), mat.bytes()});
}