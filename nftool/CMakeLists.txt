
add_executable(nftool
    nftool.cpp
    tasks.h
    run_compute.cpp
    run_render.cpp
    run_look.cpp
    run_taskcvt.cpp run_list.cpp)

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

if (${WIN32})
    DLLD_add_deploy(nftool
        BUILD_MODE
        INSTALL_MODE INSTALL_DESTINATION bin)
endif ()
