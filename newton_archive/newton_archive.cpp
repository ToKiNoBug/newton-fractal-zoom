//
// Created by joseph on 6/18/23.
//

#include "newton_archive.h"
#include <fmt/format.h>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/zstd.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <nlohmann/json.hpp>
#include <magic_enum/magic_enum.hpp>

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

  if (!os) {
    return tl::make_unexpected("The given std::ostream is not ok.");
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

  segs.emplace_back((int64_t)data_tag::map_has_result,
                    std::span<const uint8_t>{reinterpret_cast<const uint8_t *>(
                                                 this->m_map_has_result.data()),
                                             this->m_map_has_result.bytes()});
  segs.emplace_back(
      (int64_t)data_tag::map_nearest_point_idx,
      std::span<const uint8_t>{reinterpret_cast<const uint8_t *>(
                                   this->m_map_nearest_point_idx.data()),
                               this->m_map_nearest_point_idx.bytes()});
  segs.emplace_back(
      (int64_t)data_tag::map_complex_difference,
      std::span<const uint8_t>{reinterpret_cast<const uint8_t *>(
                                   this->m_map_complex_difference.data()),
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

  std::ofstream ofs{filename.data(), std::ios::binary};

  if (!ofs) {
    return tl::make_unexpected(
        fmt::format("Failed to open/create \"{}\"", filename));
  }

  auto ret = this->save(ofs);

  ofs.close();
  return ret;
}

tl::expected<void, std::string> newton_archive::save(
    std::string_view filename) const noexcept {
  if (filename.ends_with(".nfar")) {
    return this->save_raw(filename);
  }

  boost::iostreams::filtering_ostream fos;
  fos.set_auto_close(true);
  bool match{false};

  if (filename.ends_with(".nfar.zst")) {
    fos.push(boost::iostreams::zstd_compressor{});
    match = true;
  }
  if (filename.ends_with(".nfar.zlib")) {
    fos.push(boost::iostreams::zlib_compressor{});
    match = true;
  }
  if (filename.ends_with(".nfar.gz")) {
    fos.push(boost::iostreams::gzip_compressor{});
    match = true;
  }

  if (!match) {
    return tl::make_unexpected(fmt::format(
        "Failed to deduce file format from filename \"{}\"", filename));
  }
  fos.push(boost::iostreams::file_sink{filename.data(), std::ios::binary});
  auto ret = this->save(fos);

  return ret;
}

tl::expected<void, std::string> newton_archive::load(
    std::istream &is, bool ignore_compute_objects, std::span<uint8_t> buffer,
    const load_options &opt) & noexcept {
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

#define NF_PRIVATE_MARCO_LOAD_MATRIX(tag_enum_name, map_member_name)       \
  {                                                                        \
    auto blk = ar.find_first_of(int64_t(data_tag::tag_enum_name));         \
    if (blk == nullptr) {                                                  \
      return tl::make_unexpected(                                          \
          fmt::format("Failed to find data block with tag {} (aka {})",    \
                      int64_t(data_tag::tag_enum_name),                    \
                      magic_enum::enum_name(data_tag::tag_enum_name)));    \
    }                                                                      \
    if (blk->bytes() != this->map_member_name.bytes()) {                   \
      return tl::make_unexpected(fmt::format(                              \
          "Data length mismatch for {}: expected {} "                      \
          "bytes, but actually {} bytes.",                                 \
          #map_member_name, this->map_member_name.bytes(), blk->bytes())); \
    }                                                                      \
    memcpy(this->map_member_name.data(), blk->data(),                      \
           this->map_member_name.bytes());                                 \
  }

  // load map_has_result
  NF_PRIVATE_MARCO_LOAD_MATRIX(map_has_result, m_map_has_result);

  // load map_nearest_point_idx
  NF_PRIVATE_MARCO_LOAD_MATRIX(map_nearest_point_idx, m_map_nearest_point_idx);

  // load map_nearest_point_idx
  NF_PRIVATE_MARCO_LOAD_MATRIX(map_complex_difference,
                               m_map_complex_difference);

  if (opt.return_archive != nullptr) {
    *opt.return_archive = std::move(ar);
  }

  return {};
}

[[nodiscard]] size_t expected_buffer_size(int rows, int cols) noexcept {
  return 4096 +
         size_t(rows) * size_t(cols) *
             (sizeof(bool) + sizeof(uint8_t) + sizeof(std::complex<double>));
}

tl::expected<void, std::string> newton_archive::load(
    std::string_view filename, bool ignore_compute_objects,
    std::span<uint8_t> buffer, const load_options &opt) & noexcept {
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
  if (filename.ends_with(".nfar.zlib")) {
    fls.push(boost::iostreams::zlib_decompressor{});
    match = true;
  }
  if (filename.ends_with(".nfar.gz")) {
    fls.push(boost::iostreams::gzip_decompressor{});
    match = true;
  }

  if (!match) {
    return tl::make_unexpected(
        fmt::format("Failed to deduce encoding from filename {}", filename));
  }
  fls.push(boost::iostreams::file_source{filename.data(), std::ios::binary});

  return this->load(fls, ignore_compute_objects, buffer, opt);
}

tl::expected<newton_archive, std::string> newton_archive::load_archive(
    std::string_view filename, bool ignore_compute_objects,
    std::span<uint8_t> buffer, const load_options &opt) noexcept {
  newton_archive ret;

  auto temp = ret.load(filename, ignore_compute_objects, buffer, opt);
  if (!temp.has_value()) {
    return tl::make_unexpected(temp.error());
  }
  return ret;
}

tl::expected<void, std::string> check_archive(
    const newton_archive &ar) noexcept {
  if (ar.info().num_points() > 255 || ar.info().num_points() <= 0) {
    return tl::make_unexpected(fmt::format(
        "num_points should be in range [0,255], but the number of points is {}",
        ar.info().num_points()));
  }

  const uint8_t num_points = ar.info().num_points();
  for (int r = 0; r < ar.info().rows; r++) {
    for (int c = 0; c < ar.info().cols; c++) {
      if (ar.map_has_result().at<bool>(r, c)) {
        const auto val =
            ar.map_complex_difference().at<std::complex<double>>(r, c);
        if (!std::isfinite(val.real()) || !std::isfinite(val.imag())) {
          return tl::make_unexpected(fmt::format(
              "complex_difference at [{}, {}] is {}+{}i, but has_result is {}",
              r, c, val.real(), val.imag(),
              ar.map_has_result().at<bool>(r, c)));
        }
      }

      const auto npi = ar.map_nearest_point_idx().at<uint8_t>(r, c);
      if (npi >= num_points) {
        return tl::make_unexpected(
            fmt::format("nearest_point_idx at [{}, {}] is {}, but the number "
                        "of points is {}.",
                        r, c, npi, num_points));
      }
    }
  }

  return {};
}

}  // namespace newton_fractal