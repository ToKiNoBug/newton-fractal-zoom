add_library(newton_archive STATIC
        newton_archive.h
        newton_archive.cpp)

target_link_libraries(newton_archive PUBLIC
        newton_fractal)
target_include_directories(newton_archive INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS newton_archive
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)