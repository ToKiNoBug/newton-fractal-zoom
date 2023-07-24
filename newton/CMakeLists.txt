cmake_minimum_required(VERSION 3.26)

add_library(newton_fractal STATIC
    newton_fractal.h
    newton_fractal.cpp

    newton_equation_base.cpp
    computation.hpp
    newton_equation_base.h
    newton_equation.hpp

    mpc_support.h
    object_creator.cpp
    object_creator.h

    cuda_support.cu
    cuda_support.cuh
    cuda_support.cpp

    OpenCL_support.cpp
    OpenCL_support.h
    #newton_fractal_computation.cl
    )

target_compile_features(newton_fractal PUBLIC cxx_std_20)

if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    if (${MSVC})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
        #target_compile_options(newton_fractal PRIVATE "/fp:fast")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
        #target_compile_options(newton_fractal PRIVATE "-Ofast")
    endif ()
endif ()


find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(magic_enum REQUIRED)
find_package(fractal_utils COMPONENTS
    core_utils
    #png_utils
    multiprecision_utils
    REQUIRED)
find_package(OpenCL 3.0 REQUIRED)

message("OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS}")
message("OpenCL_INCLUDE_DIR = ${OpenCL_INCLUDE_DIR}")

find_file(opencl_hpp_file opencl.hpp
    PATHS "${OpenCL_INCLUDE_DIR}/CL"
    )
if (opencl_hpp_file)
    target_compile_definitions(newton_fractal PRIVATE NF_HAS_OPENCL_HPP=true)
else ()
endif ()

add_resource_config_time(cl_source_file_resource
    VAR newton_fractal_computation_cl
    newton_fractal_computation.cl)


target_link_libraries(newton_fractal PUBLIC
    fmt::fmt
    tl::expected
    fractal_utils::core_utils
    fractal_utils::multiprecision_utils
    OpenMP::OpenMP_CXX
    magic_enum::magic_enum
    OpenCL::OpenCL
    cl_source_file_resource
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

cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH cxx_compiler_parent_dir)
cmake_path(GET cxx_compiler_parent_dir PARENT_PATH cxx_compiler_parent_dir)
#message(STATUS "cxx_compiler_parent_dir = ${cxx_compiler_parent_dir}")
list(APPEND CMAKE_PREFIX_PATH ${cxx_compiler_parent_dir})
find_program(clang_exe
    clang)
if (clang_exe)
    add_custom_target(test_opencl_sources ALL
        COMMAND ${clang_exe} "${CMAKE_CURRENT_SOURCE_DIR}/newton_fractal_computation.cl" -c -emit-llvm -o newton_fractal_computation.bc
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES newton_fractal_computation.cl
        COMMENT "Build opencl sources with clang(test only)"
        BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/newton_fractal_computation.bc")
endif ()

install(TARGETS newton_fractal
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
file(GLOB compute_presets "${CMAKE_SOURCE_DIR}/compute_tasks/*.json")
install(FILES ${compute_presets} DESTINATION compute_presets)