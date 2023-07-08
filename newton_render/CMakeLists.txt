find_package(fractal_utils COMPONENTS png_utils render_utils REQUIRED)
if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)
endif ()

add_library(newton_render STATIC
    newton_render.h
    newton_render.cpp

    render_config.h
    render_config.cpp
    cpu_renderer.cpp
    cpu_renderer.h

    NF_cuda_macros.hpp
    render_config_gpu_impl.cu

    gpu_interface.h
    gpu_internal.h
    gpu_impl.cu)

target_link_libraries(newton_render PUBLIC
    #newton_fractal
    tl::expected
    magic_enum::magic_enum
    fractal_utils::core_utils
    fractal_utils::png_utils
    fractal_utils::render_utils)

target_compile_features(newton_render PUBLIC cxx_std_17)

target_include_directories(newton_render PRIVATE
    ${NF_njson_include_dir})
target_include_directories(newton_render INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_newton_render test_newton_render.cpp)
target_link_libraries(test_newton_render PRIVATE
    newton_fractal
    newton_render)
target_include_directories(test_newton_render PRIVATE
    ${NF_cli11_include_dir})


install(TARGETS newton_render
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

file(GLOB render_json_files "${CMAKE_SOURCE_DIR}/render_jsons/*.json")
install(FILES ${render_json_files} DESTINATION render_presets)