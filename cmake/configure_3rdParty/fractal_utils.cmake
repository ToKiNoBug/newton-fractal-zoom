find_package(fractal_utils QUIET)

if (${fractal_utils_FOUND})
    return()
endif ()

include(FetchContent)

if (NOT DEFINED NF_USE_QUADMATH)
    message(WARNING "NF_USE_QUADMATH ought to be defined, using default value OFF")
    set(NF_USE_QUADMATH OFF)
endif ()
set(FU_USE_QUADMATH ${NF_USE_QUADMATH})

message(STATUS "Downloading fractal_utils...")
FetchContent_Declare(fractal_utils
    GIT_REPOSITORY https://github.com/ToKiNoBug/FractalUtils.git
    GIT_TAG v2.3.27
    OVERRIDE_FIND_PACKAGE)

FetchContent_MakeAvailable(fractal_utils)

find_package(fractal_utils 2.3.27 REQUIRED)
