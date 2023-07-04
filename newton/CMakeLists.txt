cmake_minimum_required(VERSION 3.26)

add_library(newton_fractal STATIC
        newton_fractal.h
        newton_fractal.cpp

        newton_equation_base.cpp
        newton_equation_base.h
        newton_equation.hpp

        mpc_support.h
        object_creator.cpp
        object_creator.h)

target_compile_features(newton_fractal PUBLIC cxx_std_20)

if (${MSVC})
    target_compile_options(newton_fractal PRIVATE "/fp:fast")
else ()
    target_compile_options(newton_fractal PRIVATE "-Ofast")
endif ()


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

if ((${LINUX}) OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"))
    target_link_libraries(newton_fractal PUBLIC quadmath)
endif ()

if (${NF_mpc_support})
    target_sources(newton_fractal PRIVATE mpc_support.cpp)
    target_compile_definitions(newton_fractal PUBLIC NEWTON_FRACTAL_MPC_SUPPORT=1)
    target_link_libraries(newton_fractal PUBLIC mpfr mpc)
endif ()

add_executable(test_equation test_equation.cpp)
target_link_libraries(test_equation PRIVATE newton_fractal)

add_executable(test_object_creator test_object_creator.cpp)
target_link_libraries(test_object_creator PRIVATE newton_fractal)

add_executable(test_metdainfo test_metdainfo.cpp)
target_link_libraries(test_metdainfo PRIVATE newton_fractal)
target_include_directories(test_metdainfo PRIVATE ${NF_cli11_include_dir})


install(TARGETS newton_fractal
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
file(GLOB compute_presets "${CMAKE_SOURCE_DIR}/compute_tasks/*.json")
install(FILES ${compute_presets} DESTINATION compute_presets)