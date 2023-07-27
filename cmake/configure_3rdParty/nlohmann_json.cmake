
set(NF_njson_include_dir "${CMAKE_BINARY_DIR}/3rdParty")
set(NF_njson_file ${NF_njson_include_dir}/nlohmann/json.hpp)

if (EXISTS ${NF_njson_file})
    return()
endif ()

message(STATUS "Downloading nlohmann json...")
#file(DOWNLOAD )
NF_Download(https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp ${NF_njson_file})