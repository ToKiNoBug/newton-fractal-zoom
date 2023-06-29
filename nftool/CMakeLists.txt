
add_executable(nftool
        nftool.cpp
        run_compute.h
        run_compute.cpp
        run_render.cpp
        run_look.cpp)

target_link_libraries(nftool PRIVATE
        newton_fractal
        newton_archive
        newton_render
        )

target_include_directories(nftool PRIVATE
        ${NF_cli11_include_dir}
        ${NF_njson_include_dir})


install(TARGETS nftool
        RUNTIME DESTINATION bin)