cmake_minimum_required(VERSION 3.26)
project(newton_fractal_zoom VERSION 0.3.0)

set(CMAKE_CXX_EXTENSIONS OFF)

if (${MSVC})
    #list(APPEND CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "/EHsc /wd4819")

    #add_compile_options()
endif ()

#add_compile_options(-pg)
#add_link_options(-pg)


include(cmake/configure_3rdParty.cmake)

find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(fractal_utils 2.3.21 COMPONENTS
    core_utils
    png_utils
    multiprecision_utils
    render_utils
    REQUIRED)
find_package(tl-expected REQUIRED)


set(NF_mpc_support OFF)
find_package(MPC QUIET)
if (MPC_FOUND OR LINUX)
    set(NF_mpc_support ON)
endif ()

#if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
#    set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
#    message(STATUS "This project will be compiled with clang, so clang is used instead of nvcc to compile cuda source codes.")
#endif ()
enable_language(CUDA)
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
set(CMAKE_CUDA_STANDARD 20)
if (${MSVC})
    set(CMAKE_CUDA_FLAGS "/wd4819")
endif ()
if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)
endif ()
#set(CMAKE_CLANG)

add_compile_definitions("NEWTON_FRACTAL_VERSION_STR=\"${PROJECT_VERSION}\"")

enable_testing()

add_subdirectory(newton)
add_subdirectory(newton_archive)
add_subdirectory(newton_render)
add_subdirectory(nftool)
add_subdirectory(newton_zoomer)
add_subdirectory(nfvideo)

include(cmake/pack.cmake)