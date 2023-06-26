find_package(fractal_utils QUIET)

if (${fractal_utils_FOUND})
    return()
endif ()

include(FetchContent)

message(STATUS "Downloading fractal_utils...")
FetchContent_Declare(fractal_utils
        GIT_REPOSITORY https://github.com/ToKiNoBug/FractalUtils.git
        GIT_TAG v2.3.8
        OVERRIDE_FIND_PACKAGE)

FetchContent_MakeAvailable(fractal_utils)

find_package(fractal_utils REQUIRED)
