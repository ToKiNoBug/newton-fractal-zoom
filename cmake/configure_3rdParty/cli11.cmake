
set(NF_cli11_include_dir "${CMAKE_BINARY_DIR}/3rdParty/cli11")
set(NF_cli11_file ${NF_cli11_include_dir}/CLI11.hpp)

if (EXISTS ${NF_cli11_file})
    return()
endif ()

message(STATUS "Downloading CLI11.hpp...")
file(DOWNLOAD https://github.com/CLIUtils/CLI11/releases/download/v2.3.2/CLI11.hpp ${NF_cli11_file})