
set(NF_cli11_include_dir "${CMAKE_BINARY_DIR}/3rdParty/cli11")
set(NF_cli11_file ${NF_cli11_include_dir}/CLI11.hpp)

if (EXISTS ${NF_cli11_file})

    file(SIZE ${NF_cli11_file} file_size)

    if (${file_size} LESS_EQUAL 0)
        file(REMOVE ${NF_cli11_file})
    else ()
        return()
    endif ()

endif ()

message(STATUS "Downloading CLI11.hpp...")
file(DOWNLOAD https://github.com/CLIUtils/CLI11/releases/download/v2.3.2/CLI11.hpp ${NF_cli11_file})