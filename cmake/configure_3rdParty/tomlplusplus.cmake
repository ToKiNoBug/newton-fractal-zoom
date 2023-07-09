#https://github.com/marzer/tomlplusplus
find_package(tomlplusplus QUIET)
if (${tomlplusplus_FOUND})
    return()
endif ()

message(STATUS "Downloading tomlplusplus...")
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.3.0
    OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(tomlplusplus)
