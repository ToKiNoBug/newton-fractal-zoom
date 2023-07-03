find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(fractal_utils COMPONENTS zoom_utils REQUIRED)

add_executable(nfzoom
        nfzoom.cpp
        newton_label.h
        newton_label.cpp newton_zoomer.cpp newton_zoomer.h)
target_link_libraries(nfzoom PRIVATE
        newton_archive
        newton_render
        Qt6::Widgets
        fractal_utils::zoom_utils)
target_include_directories(nfzoom PRIVATE
        ${NF_cli11_include_dir})