#include <CLI11.hpp>
#include "load_video_task.h"
#include "video_executor.h"

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

  return 0;
}