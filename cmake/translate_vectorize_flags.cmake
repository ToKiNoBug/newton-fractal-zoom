function(NF_translate_vec_flags IS_list flags_out)
    unset(${flags_out})

    set(IS_LUT "AVX;AVX2;FMA")
    set(gnu_flags "-mavx;-mavx2;-mfma")
    set(msvc_flags "/arch:AVX;/arch:AVX2;")


    set(result "")
    foreach (IS ${${IS_list}})
        list(FIND IS_LUT ${IS} index)
        #message("index = ${index}")
        if (${index} LESS 0)
            message(WARNING "Invalid instruction set name ${IS}. Supported values: ${IS_LUT}")
            continue()
        endif ()

        if (${MSVC})
            list(GET msvc_flags ${index} flag)
        else ()
            list(GET gnu_flags ${index} flag)
        endif ()
        #message(STATUS "flag = ${flag}")

        set(result "${result} ${flag}")
    endforeach ()

    set(${flags_out} ${result} PARENT_SCOPE)
endfunction()