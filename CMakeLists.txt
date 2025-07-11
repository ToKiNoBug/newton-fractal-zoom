cmake_minimum_required(VERSION 3.26)
project(newton_fractal_zoom VERSION 0.6.2)


set(CMAKE_CXX_EXTENSIONS OFF)

if (${MSVC})
    #list(APPEND CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "/EHsc /wd4819")

    #add_compile_options()
endif ()

#add_compile_options(-pg)
#add_link_options(-pg)

option(NF_USE_QUADMATH "Enable GNU quadmath" OFF)
option(NF_USE_CUDA "Enable cuda support" ON)
set(NF_VECTORIZE_IS "FMA;AVX;AVX2" CACHE STRING "Instruction sets used by vectorization")

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
if (${NF_USE_CUDA})
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_STANDARD 20)
    if (${MSVC})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
    endif ()
    if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    endif ()
    add_compile_definitions("NF_USE_CUDA=1")
endif ()
#set(CMAKE_CLANG)

add_compile_definitions("NEWTON_FRACTAL_VERSION_STR=\"${PROJECT_VERSION}\"")

enable_testing()

include(cmake/translate_vectorize_flags.cmake)

NF_translate_vec_flags(NF_VECTORIZE_IS vec_flags)
message("Vectorization flags: ${vec_flags}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${vec_flags}")

add_subdirectory(newton)
add_subdirectory(newton_archive)
add_subdirectory(newton_render)
add_subdirectory(nftool)
add_subdirectory(newton_zoomer)
add_subdirectory(nfvideo)

include(cmake/pack.cmake)