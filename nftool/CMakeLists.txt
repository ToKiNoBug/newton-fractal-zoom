
add_executable(nftool nftool.cpp)
target_link_libraries(nftool PRIVATE newton_fractal)

target_include_directories(nftool PRIVATE
        ${NF_cli11_include_dir}
        ${NF_njson_include_dir})