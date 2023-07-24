#https://github.com/SlopeCraft/ResourceCreator.cmake/releases/download/v0.0.0/ResourceCreator.cmake

set(filename "${CMAKE_BINARY_DIR}/3rdParty/ResourceCreator.cmake/ResourceCreator.cmake")
NF_Download(https://github.com/SlopeCraft/ResourceCreator.cmake/releases/download/v0.0.0/ResourceCreator.cmake ${filename})

include(${filename})