if (NOT ${WIN32})
    return()
endif ()

set(DLLD_dir "${CMAKE_BINARY_DIR}/3rdParty/DLLDeployer")

set(DLLD_file ${DLLD_dir}/DLLDeployer.cmake)
set(QD_file ${DLLD_dir}/QtDeployer.cmake)


#file(REMOVE ${DLLD_file} ${QD_file})
file(DOWNLOAD https://github.com/ToKiNoBug/DLLDeployer/releases/download/v1.1/DLLDeployer.cmake
    ${DLLD_file} SHOW_PROGRESS)
file(DOWNLOAD https://github.com/ToKiNoBug/DLLDeployer/releases/download/v1.1/QtDeployer.cmake
    ${QD_file} SHOW_PROGRESS)

file(SIZE ${DLLD_file} size)
if (${size} LESS_EQUAL 0)
    message(FATAL_ERROR "Failed to download ${DLLD_file}")
endif ()
file(SIZE ${QD_file} size)
if (${size} LESS_EQUAL 0)
    message(FATAL_ERROR "Failed to download ${QD_file}")
endif ()


include(${DLLD_file})
include(${QD_file})