//
// Created by joseph on 7/24/23.
//
#include <fmt/format.h>
#include "tasks.h"
#include "OpenCL_support.h"

tl::expected<void, std::string> list_opencl() noexcept;

tl::expected<void, std::string> run_list(const list_task& lt) noexcept {
  return list_opencl();
}

tl::expected<void, std::string> list_opencl() noexcept {
  auto platforms = newton_fractal::opencl_platforms();

  for (size_t pid = 0; pid < platforms.size(); pid++) {
    fmt::print("Platform {}: {}\n", pid, platforms[pid]);

    auto devices = newton_fractal::opencl_devices(pid);
    for (size_t did = 0; did < devices.size(); did++) {
      fmt::print("  Device {}: {}\n", did, devices[did]);
    }
  }
  return {};
}