find_package(tl-expected QUIET)

if (tl-expected_FOUND)
    return()
endif ()

message(STATUS "Downloading tl-expected...")
FetchContent_Declare(tl-expected
        GIT_REPOSITORY https://github.com/TartanLlama/expected.git
        GIT_TAG v1.1.0
        OVERRIDE_FIND_PACKAGE)

FetchContent_MakeAvailable(tl-expected)

find_package(tl-expected REQUIRED)
