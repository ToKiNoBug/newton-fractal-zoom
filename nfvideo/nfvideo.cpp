#include <CLI11.hpp>
#include "load_video_task.h"
#include "video_executor.h"
#include <fmt/format.h>
#include <toml++/toml.h>

int main(int argc, char** argv) {
  CLI::App app;
  std::string toml_file{"nfvideo.toml"};
  app.add_option("task file", toml_file)
      ->check(CLI::ExistingFile)
      ->check(CLI::Validator{[](std::string& in) -> std::string {
                               if (in.ends_with(".toml")) return {};
                               return "Extension must be .toml";
                             },
                             "Extension must be .toml", "Extension check"});

  CLI11_PARSE(app, argc, argv);

  try {
    auto result = toml::parse_file(toml_file);
    auto common = result.at("common").as_table();
    fmt::print("common.start_task_file = \"{}\"\n",
               common->at("start_task_file").value<std::string>().value());
    fmt::print("common.archive_num = {}\n",
               common->at("archive_num").value<int>().value());
    auto ratio = common->at("ratio").value<double>().value();
    fmt::print("common.ratio = {}\n", ratio);

  } catch (const std::exception& e) {
    fmt::print("Exception: {}", e.what());
  }

  return 0;
}