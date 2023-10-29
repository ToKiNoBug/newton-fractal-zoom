#include <video_utils.h>
#include "video_executor.h"
#include <toml++/toml.h>
#include <fmt/format.h>
#include <newton_archive.h>
#include <thread>

std::unique_ptr<fu::common_info_base> video_executor::load_common_info(
    std::string &err) const noexcept {
  abort();
}

std::unique_ptr<fu::compute_task_base> video_executor::load_compute_task(
    std::string &err) const noexcept {
  abort();
}

std::unique_ptr<fu::render_task_base> video_executor::load_render_task(
    std::string &err) const noexcept {
  abort();
}

std::unique_ptr<fu::video_task_base> video_executor::load_video_task(
    std::string &err) const noexcept {
  abort();
}

common_info parse_ci(const toml::table *tbl) noexcept(false);

compute_task parse_ct(const toml::table *tbl) noexcept(false);

render_task parse_rt(const toml::table *tbl) noexcept(false);

video_task parse_vt(const toml::table *tbl) noexcept(false);

std::optional<fu::full_task> video_executor::load_task(
    std::string &err) const noexcept {
  fu::full_task ret;
  try {
    toml::table task = toml::parse_file(this->task_file);

    ret.common =
        std::make_unique<common_info>(parse_ci(task.at("common").as_table()));
    assert(ret.common != nullptr);
    {
      auto compute_temp = parse_ct(task.at("compute").as_table());
      compute_temp.related_ci = dynamic_cast<common_info *>(ret.common.get());
      ret.compute = std::make_unique<compute_task>(std::move(compute_temp));
    }
    assert(ret.compute != nullptr);
    ret.render =
        std::make_unique<render_task>(parse_rt(task.at("render").as_table()));
    assert(ret.render != nullptr);

    ret.video =
        std::make_unique<video_task>(parse_vt(task.at("video").as_table()));
    assert(ret.video != nullptr);
    // dynamic_cast<compute_task *>(ret.compute.get())->related_ci =

  } catch (const toml::parse_error &e) {
    fmt::print("Failed to parse the task file: {}", e.what());
    return std::nullopt;
  } catch (const std::exception &e) {
    fmt::print("Exception occurred when parsing task file: {}", e.what());
    return std::nullopt;
  } catch (...) {
    fmt::print("Unknown exception caught.");
    return std::nullopt;
  }

  return ret;
}

void throw_if_null(const toml::table *tbl) noexcept(false) {
  if (tbl == nullptr) {
    throw std::runtime_error{"table is nullptr"};
  }
}

common_info parse_ci(const toml::table *tbl) noexcept(false) {
  throw_if_null(tbl);
  common_info ret;
  {
    auto metadata = nf::load_metadata_from_file(
        tbl->at("start_task_file").value<std::string_view>().value(), false);
    if (!metadata.has_value()) {
      throw std::runtime_error{metadata.error()};
    }
    ret.metadata = std::move(metadata.value());
  }

  ret.archive_num = tbl->at("archive_num").value<int>().value();
  ret.ratio = tbl->at("ratio").value<double>().value();
  return ret;
}

compute_task parse_ct(const toml::table *tbl) noexcept(false) {
  throw_if_null(tbl);

  compute_task ct;
  ct.threads = tbl->at("threads").value<int>().value_or(
      std::thread::hardware_concurrency());
  ct.archive_prefix = tbl->at("archive_prefix").value<std::string>().value();
  ct.archive_suffix = tbl->at("archive_suffix").value<std::string>().value();
  ct.archive_extension =
      tbl->at("archive_extension").value<std::string>().value();
  if (tbl->contains("no_check_frames")) {
    auto arrp = tbl->at("no_check_frames").as_array();
    if (arrp == nullptr) {
      throw std::runtime_error{
          fmt::format("no_check_frames\" should be array.")};
    }
    for (auto &val : *arrp) {
      ct.no_check_frames.emplace(val.value<int>().value());
    }
  }
  return ct;
}

render_task parse_rt(const toml::table *tbl) noexcept(false) {
  throw_if_null(tbl);
  render_task rt;
  rt.image_per_frame = tbl->at("image_per_frame").value<int>().value();
  rt.extra_image_num = tbl->at("extra_image_num").value<int>().value();
  rt.threads = tbl->at("threads").value<int>().value_or(
      std::thread::hardware_concurrency());
  rt.render_once = tbl->at("render_once").value<bool>().value();
  rt.image_prefix = tbl->at("image_prefix").value<std::string>().value();
  rt.image_suffix = tbl->at("image_suffix").value<std::string>().value();
  rt.image_extension = "png";
  // rt.image_extension=
  // tbl->at("image_extension").value<std::string>().value();
  {
    auto config_exp = nf::load_render_config_from_file(
        tbl->at("render_json_file").value<std::string>().value());
    if (!config_exp) {
      throw std::runtime_error{
          fmt::format("Failed to load render json file \"{}\" because {}",
                      tbl->at("render_json_file").value<std::string>().value(),
                      config_exp.error())};
    }
    rt.render_config = std::move(config_exp.value());
  }
  return rt;
}

video_task::video_config parse_vt_config(const toml::table *tbl) noexcept(
    false) {
  throw_if_null(tbl);
  video_task::video_config ret;
  ret.video_prefix = tbl->at("video_prefix").value<std::string>().value();
  ret.video_suffix = tbl->at("video_suffix").value<std::string>().value();
  ret.encoder_flags =
      tbl->at("encoder_flags").value<std::string>().value_or("");
  ret.encoder = tbl->at("encoder").value<std::string>().value_or("x264");
  ret.extension = tbl->at("extension").value<std::string>().value_or("mp4");
  return ret;
}

video_task parse_vt(const toml::table *tbl) noexcept(false) {
  throw_if_null(tbl);
  video_task vt;

  vt.temp_config = parse_vt_config(tbl->at("temp").as_table());
  vt.product_config = parse_vt_config(tbl->at("product").as_table());
  vt.threads = tbl->at("threads").value<int>().value_or(4);
  if (tbl->contains("ffmpeg_exe")) {
    vt.ffmpeg_exe = tbl->at("ffmpeg_exe").value<std::string>().value();
  } else {
    vt.ffmpeg_exe = "ffmpeg";
  }

  vt.product_name = tbl->at("product_name").value<std::string>().value();

  return vt;
}