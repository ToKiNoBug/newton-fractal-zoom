cmake_minimum_required(VERSION 3.26)
project(newton_fractal_zoom)

set(CMAKE_CXX_EXTENSIONS OFF)

if (${MSVC})
    add_compile_options("/EHsc")
endif ()


include(cmake/configure_3rdParty.cmake)

find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(fractal_utils 2.3.2 COMPONENTS
        core_utils
        png_utils
        multiprecision_utils
        REQUIRED)
find_package(tl-expected REQUIRED)


set(NF_mpc_support OFF)
find_package(MPC QUIET)
if (MPC_FOUND OR LINUX)
    set(NF_mpc_support ON)
endif ()

add_subdirectory(newton)
add_subdirectory(newton_archive)

enable_language(CUDA)
add_subdirectory(nftool)
