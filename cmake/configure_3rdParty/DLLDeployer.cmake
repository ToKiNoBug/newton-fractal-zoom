if (NOT ${WIN32})
    return()
endif ()

set(DLLD_dir "${CMAKE_BINARY_DIR}/3rdParty/DLLDeployer")

set(DLLD_file ${DLLD_dir}/DLLDeployer.cmake)
set(QD_file ${DLLD_dir}/QtDeployer.cmake)

file(DOWNLOAD https://github.com/ToKiNoBug/DLLDeployer/releases/download/v1.1/DLLDeployer.cmake
    ${DLLD_file})
file(DOWNLOAD https://github.com/ToKiNoBug/DLLDeployer/releases/download/v1.1/QtDeployer.cmake
    ${QD_file})

include(${DLLD_file})
include(${QD_file})