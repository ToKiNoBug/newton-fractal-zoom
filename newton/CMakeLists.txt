cmake_minimum_required(VERSION 3.26)

add_library(newton_fractal STATIC
        newton_fractal.h
        newton_fractal.cpp)

target_compile_features(newton_fractal PUBLIC cxx_std_20)


find_package(fmt REQUIRED)
find_package(fractal_utils COMPONENTS
        core_utils
        #png_utils
        multiprecision_utils
        REQUIRED)

target_link_libraries(newton_fractal PUBLIC
        fmt::fmt
        tl::expected
        fractal_utils::core_utils
        fractal_utils::multiprecision_utils
        
        )


add_executable(test_equation test_equation.cpp)
target_link_libraries(test_equation PRIVATE newton_fractal)

if (${NF_mpc_support})
    target_compile_definitions(newton_fractal PUBLIC NEWTON_FRACTAL_MPC_SUPPORT=1)
    target_link_libraries(newton_fractal PUBLIC mpfr mpc)
endif ()
