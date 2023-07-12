find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(fractal_utils COMPONENTS zoom_utils REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

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

add_executable(nfzoom_launcher nfzoom_launcher.cpp launcher_wind.cpp launcher_wind.h launcher_wind.ui)
target_link_libraries(nfzoom_launcher PRIVATE
    tl::expected
    Qt6::Widgets)
file(GLOB compute_presets "${CMAKE_SOURCE_DIR}/compute_tasks/*.json")
file(COPY ${compute_presets} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../compute_presets")
file(GLOB render_json_files "${CMAKE_SOURCE_DIR}/render_jsons/*.json")
file(COPY ${render_json_files} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../render_presets")

install(TARGETS nfzoom RUNTIME DESTINATION bin)
install(TARGETS nfzoom_launcher RUNTIME DESTINATION bin)


if (${WIN32})
    QD_add_deployqt(nfzoom
        BUILD_MODE
        INSTALL_MODE INSTALL_DESTINATION bin
        FLAGS -no-translations
        )
    DLLD_add_deploy(nfzoom
        BUILD_MODE
        INSTALL_MODE INSTALL_DESTINATION bin)


    QD_add_deployqt(nfzoom_launcher
        BUILD_MODE
        INSTALL_MODE INSTALL_DESTINATION bin
        FLAGS -no-translations
        )
    DLLD_add_deploy(nfzoom_launcher
        BUILD_MODE
        INSTALL_MODE INSTALL_DESTINATION bin)
endif ()