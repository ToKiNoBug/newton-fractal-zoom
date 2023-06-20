//
// Created by joseph on 6/18/23.
//

#include "newton_archive.h"
#include <fmt/format.h>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/zstd.hpp>
#include <zstd.h>
#include <filesystem>
#include <magic_enum.hpp>

namespace stdfs = std::filesystem;

namespace newton_fractal {

newton_archive::newton_archive(const meta_data &mdsrc) noexcept
    : m_info{mdsrc} {
  this->setup_matrix();
}

newton_archive::newton_archive(meta_data &&mdsrc) noexcept : m_info{mdsrc} {
  this->setup_matrix();
}

void newton_archive::setup_matrix() & noexcept {
  this->m_map_has_result.reset(this->m_info.rows, this->m_info.cols,
                               sizeof(bool));
  this->m_map_nearest_point_idx.reset(this->m_info.rows, this->m_info.cols,
                                      sizeof(uint8_t));
  this->m_map_complex_difference.reset(this->m_info.rows, this->m_info.cols,
                                       sizeof(std::complex<double>));
}

tl::expected<void, std::string> newton_archive::save(
    std::ostream &os) const noexcept {
  if (!this->info().can_compute()) {
    return tl::make_unexpected(
        "The compute objects are missing from metainfo struct.");
  }

  fu::binary_archive ar;

  {
    fu::file_header fh;
    memset(fh.custom_part(), 0, fu::file_header::custom_part_len());
    ar.set_header(fh);
  }

  auto &segs = ar.segments();
  segs.reserve(4);
  segs.clear();

  std::string json_str;
  {
    auto jo_e = save_metadata(this->m_info);
    if (!jo_e.has_value()) {
      return tl::make_unexpected(fmt::format(
          "Failed to serialize metadata to json, detail: {}", jo_e.error()));
    }
    json_str = jo_e.value().dump();

    std::span<uint8_t> span{reinterpret_cast<uint8_t *>(json_str.data()),
                            json_str.size()};
    segs.emplace_back((int64_t)data_tag::metadata, span);
  }

  segs.emplace_back(
      (int64_t)data_tag::map_has_result,
      std::span<uint8_t>{reinterpret_cast<uint8_t *>(
                             const_cast<void *>(this->m_map_has_result.data())),
                         this->m_map_has_result.bytes()});
  segs.emplace_back(
      (int64_t)data_tag::map_nearest_point_idx,
      std::span<uint8_t>{reinterpret_cast<uint8_t *>(const_cast<void *>(
                             this->m_map_nearest_point_idx.data())),
                         this->m_map_nearest_point_idx.bytes()});
  segs.emplace_back(
      (int64_t)data_tag::map_complex_difference,
      std::span<uint8_t>{reinterpret_cast<uint8_t *>(const_cast<void *>(
                             this->m_map_complex_difference.data())),
                         this->m_map_complex_difference.bytes()});

  auto err = ar.save(os);

  if (!err.empty()) {
    return tl::make_unexpected(
        fmt::format("binary_archive::save failed with error info : {}", err));
  }

  return {};
}

tl::expected<void, std::string> newton_archive::save_raw(
    std::string_view filename) const noexcept {
  if (!filename.ends_with(".nfar")) {
    return tl::make_unexpected(fmt::format(
        "The file extension must be .nfar, but the given filename is {}",
        filename));
  }

  std::ofstream ofs{filename.data()};

  if (!ofs) {
    return tl::make_unexpected(
        fmt::format("Failed to open/create \"{}\"", filename));
  }

  auto ret = this->save(ofs);

  ofs.close();
  return ret;
}
tl::expected<void, std::string> newton_archive::save_zstd(
    std::string_view filename) const noexcept {
  if (!filename.ends_with(".nfar.zst")) {
    return tl::make_unexpected(fmt::format(
        "The file extension must be .nfar.zst, but the given filename is {}",
        filename));
  }
  boost::iostreams::filtering_ostream fos;
  fos.push(boost::iostreams::zstd_compressor{});
  fos.push(boost::iostreams::file_sink(filename.data()));

  return this->save(fos);
}

tl::expected<void, std::string> newton_archive::save(
    std::string_view filename) const noexcept {
  if (filename.ends_with(".nfar")) {
    return this->save_raw(filename);
  }

  if (filename.ends_with(".nfar.zst")) {
    return this->save_zstd(filename);
  }

  return tl::make_unexpected(fmt::format(
      "Failed to deduce file format from filename \"{}\"", filename));
}

tl::expected<void, std::string> newton_archive::load(
    std::istream &is, bool ignore_compute_objects,
    std::span<uint8_t> buffer) & noexcept {
  fu::binary_archive ar;

  auto err = ar.load(is, buffer, nullptr);
  if (!err.empty()) {
    return tl::make_unexpected(
        fmt::format("Failed to load binary archive, detail: {}", err));
  }

  // load meta info
  {
    auto blk = ar.find_first_of(int64_t(data_tag::metadata));
    if (blk == nullptr) {
      return tl::make_unexpected(
          fmt::format("Failed to find data block with tag {} (aka {})",
                      int64_t(data_tag::metadata),
                      magic_enum::enum_name(data_tag::metadata)));
    }

    std::string_view json{(const char *)blk->data(), blk->bytes()};

    auto mi = load_metadata(json, ignore_compute_objects);
    if (!mi.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to parse metadata, detail: {}", mi.error()));
    }
    this->m_info = std::move(mi.value());
  }
  this->setup_matrix();

#define NF_PRIVATE_MARCO_LOAD_MATRIX(tag_enum_name, map_member_name)        \
  {                                                                         \
    auto blk = ar.find_first_of(int64_t(data_tag::tag_enum_name));          \
    if (blk == nullptr) {                                                   \
      return tl::make_unexpected(                                           \
          fmt::format("Failed to find data block with tag {} (aka {})",     \
                      int64_t(data_tag::tag_enum_name),                     \
                      magic_enum::enum_name(data_tag::tag_enum_name)));     \
    }                                                                       \
    if (blk->bytes() != this->m_map_has_result.bytes()) {                   \
      return tl::make_unexpected(fmt::format(                               \
          "Data length mismatch for {}: expected {} "                       \
          "bytes, but actually {} bytes.",                                  \
          #map_member_name, this->m_map_has_result.bytes(), blk->bytes())); \
    }                                                                       \
    memcpy(this->map_member_name.data(), blk->data(),                       \
           this->map_member_name.bytes());                                  \
  }

  // load map_has_result
  NF_PRIVATE_MARCO_LOAD_MATRIX(map_has_result, m_map_has_result);

  // load map_nearest_point_idx
  NF_PRIVATE_MARCO_LOAD_MATRIX(map_nearest_point_idx, m_map_nearest_point_idx);

  // load map_nearest_point_idx
  NF_PRIVATE_MARCO_LOAD_MATRIX(map_complex_difference,
                               m_map_complex_difference);

  return {};
}

[[nodiscard]] size_t expected_buffer_size(int rows, int cols) noexcept {
  return 4096 +
         size_t(rows) * size_t(cols) *
             (sizeof(bool) + sizeof(uint8_t) + sizeof(std::complex<double>));
}

tl::expected<void, std::string> newton_archive::load(
    std::string_view filename, bool ignore_compute_objects,
    std::span<uint8_t> buffer) & noexcept {
  if (filename.ends_with(".nfar")) {
    std::ifstream ifs{filename.data()};
    return this->load(ifs, ignore_compute_objects, buffer);
  }

  boost::iostreams::filtering_istream fls;
  bool match = false;
  if (filename.ends_with(".nfar.zst")) {
    fls.push(boost::iostreams::zstd_decompressor{});
    match = true;
  }

  if (!match) {
    return tl::make_unexpected(
        fmt::format("Failed to deduce encoding from filename {}", filename));
  }
  fls.push(boost::iostreams::file_source(filename.data()));

  return this->load(fls, ignore_compute_objects, buffer);
}

}  // namespace newton_fractal