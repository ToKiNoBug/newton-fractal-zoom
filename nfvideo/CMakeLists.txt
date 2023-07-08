find_package(fractal_utils COMPONENTS video_utils)

add_executable(nfvideo
    nfvideo.cpp
    load_video_task.h
    load_video_task.cpp video_executor.cpp video_executor.h)
target_link_libraries(nfvideo PRIVATE
    fractal_utils::video_utils
    newton_render
    newton_archive)