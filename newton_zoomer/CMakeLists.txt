find_package(Qt6 COMPONENTS Widgets LinguistTools REQUIRED)
find_package(fractal_utils COMPONENTS zoom_utils REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(nfzoom_sources
    nfzoom.cpp

    newton_label.h
    newton_label.cpp

    newton_zoomer.h
    newton_zoomer.cpp

    point_form.cpp
    point_form.h
    point_form.ui
    zoomer_custom_widget.cpp
    zoomer_custom_widget.h
    zoomer_custom_widget.ui)

add_executable(nfzoom
    ${nfzoom_sources}
)
qt_add_translations(nfzoom
    SOURCES ${nfzoom_sources}
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
    TS_FILES others/nfzoom_zh_CN.ts
    OUTPUT_TARGETS nfzoom_translation_targets
    LUPDATE_OPTIONS -no-obsolete -locations relative -no-recursive
)
target_link_libraries(nfzoom PRIVATE
    newton_archive
    newton_render
    Qt6::Widgets
    fractal_utils::zoom_utils)
target_include_directories(nfzoom PRIVATE
    ${NF_cli11_include_dir})

set(nfzoom_launcher_sources
    launcher_wind.h
    launcher_wind.ui
    launcher_wind.cpp
    nfzoom_launcher.cpp)
add_executable(nfzoom_launcher
    ${nfzoom_launcher_sources})
qt_add_translations(nfzoom_launcher
    SOURCES ${nfzoom_launcher_sources}
    TS_FILES others/nfzoom_launcher_zh_CN.ts
    OUTPUT_TARGETS nfzoom_launcher_translation_targets
    LUPDATE_OPTIONS -no-obsolete -locations relative
)

target_link_libraries(nfzoom_launcher PRIVATE
    tl::expected
    Qt6::Widgets)
set_target_properties(nfzoom_launcher PROPERTIES
    WIN32_EXECUTABLE ON)
add_dependencies(nfzoom_launcher nfzoom)


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