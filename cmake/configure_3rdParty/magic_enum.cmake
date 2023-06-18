find_package(magic_enum QUIET)

if (${magic_enum_FOUND})
    return()
endif ()

message(STATUS "Downloading magic_enum...")
FetchContent_Declare(magic_enum
        GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
        GIT_TAG v0.9.2
        OVERRIDE_FIND_PACKAGE)

FetchContent_MakeAvailable(magic_enum)