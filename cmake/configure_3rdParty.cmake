include(FetchContent)

file(GLOB configure_files "${CMAKE_SOURCE_DIR}/cmake/configure_3rdParty/*.cmake")

foreach (file ${configure_files})
    include(${file})
endforeach ()
