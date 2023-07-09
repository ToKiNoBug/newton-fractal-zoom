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


set(example_task_file "${CMAKE_SOURCE_DIR}/compute_tasks/double-p3.json")
set(example_render_json_file "${CMAKE_SOURCE_DIR}/render_jsons/plasma-9.json")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/task_example.toml
        ${CMAKE_CURRENT_BINARY_DIR}/nfvideo.toml
        @ONLY)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/archive)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/image)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/video)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/video/temp)