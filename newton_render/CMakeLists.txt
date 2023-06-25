find_package(fractal_utils COMPONENTS png_utils render_utils REQUIRED)

set(CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)

add_library(newton_render STATIC
        newton_render.h
        newton_render.cpp


        gpu_interface.h
        gpu_internal.h
        gpu_interface.cu
        render_config_gpu_interface.cu render_config.h NF_cuda_macros.hpp render_config.cpp)

target_link_libraries(newton_render PUBLIC
        newton_fractal
        fractal_utils::png_utils
        fractal_utils::render_utils)

target_include_directories(newton_render INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})