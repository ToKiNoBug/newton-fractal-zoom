include(FetchContent)


function(NF_Download url filename)
    if (EXISTS ${filename})
        file(SIZE ${filename} size)
        if (${size} GREATER 0)
            message(STATUS "${filename} already exists, skip downloading.")
            return()
        endif ()
    endif ()

    message(STATUS "Downloading ${filename} ...")
    file(DOWNLOAD ${url} ${filename} SHOW_PROGRESS)

    file(SIZE ${filename} size)
    if (${size} LESS_EQUAL 0)
        message(FATAL_ERROR "Failed to download ${filename}")
    endif ()
endfunction(NF_Download)

file(GLOB configure_files "${CMAKE_SOURCE_DIR}/cmake/configure_3rdParty/*.cmake")
foreach (file ${configure_files})
    include(${file})
endforeach ()
