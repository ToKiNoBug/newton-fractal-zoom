find_package(fractal_utils
    COMPONENTS video_utils
    REQUIRED)
find_package(tomlplusplus REQUIRED)

add_executable(nfvideo
    nfvideo.cpp
    load_video_task.h
    load_video_task.cpp
    video_executor.h
    video_executor.cpp
    )

target_link_libraries(nfvideo PRIVATE
    fractal_utils::video_utils
    tomlplusplus::tomlplusplus
    newton_render
    newton_archive)

target_include_directories(nfvideo PRIVATE
    ${NF_cli11_include_dir})