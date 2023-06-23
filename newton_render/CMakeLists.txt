find_package(fractal_utils COMPONENTS png_utils render_utils REQUIRED)

set(CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)

add_library(newton_render STATIC
        newton_render.h
        newton_render.cpp


        gpu_interface.h
        gpu_internal.h
        gpu_interface.cu
        render_config_gpu_interface.cu)

target_link_libraries(newton_render PUBLIC
        newton_fractal
        fractal_utils::png_utils
        fractal_utils::render_utils)