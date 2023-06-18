cmake_minimum_required(VERSION 3.26)

add_library(newton_fractal STATIC
        newton_fractal.h
        newton_fractal.cpp

        newton_equation_base.cpp
        newton_equation_base.h
        newton_equation.hpp

        mpc_support.h newton_archive.cpp newton_archive.h)

target_compile_features(newton_fractal PUBLIC cxx_std_20)


find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(magic_enum REQUIRED)
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
        OpenMP::OpenMP_CXX
        magic_enum::magic_enum
        )

target_include_directories(newton_fractal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(newton_fractal PUBLIC
        ${NF_cli11_include_dir}
        ${NF_njson_include_dir})

add_executable(test_equation test_equation.cpp)
target_link_libraries(test_equation PRIVATE newton_fractal)

if (${NF_mpc_support})
    target_sources(newton_fractal PRIVATE mpc_support.cpp)
    target_compile_definitions(newton_fractal PUBLIC NEWTON_FRACTAL_MPC_SUPPORT=1)
    target_link_libraries(newton_fractal PUBLIC mpfr mpc)
endif ()
