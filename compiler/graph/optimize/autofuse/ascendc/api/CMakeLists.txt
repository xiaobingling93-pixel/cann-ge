set(ascendc_api_extend_src
    utils.h
    reduce.h
    reduce_prod.h
    reduce_any.h
    duplicate.h
    broadcast.h
    compare.h
    compare_v2.h
    clipbyvalue.h
    pow.h
    gelu.h
    where.h
    concat.h
    sigmoid.h
    scalar_div.h
    rsqrt.h
    removepad.h
    logical_not.h
    logical.h
    isnan.h
    sign.h
    datacopy.h
    isfinite.h
    reduce_init.h
    cast.h
    reciprocal.h
    transpose_base_type.h
    transpose.h
    bitwise_and.h
    gather.h
    scalar_sub.h
    floor_div.h
    brc_inline_api.h
    axpy.h
    scalar_add.h
    scalar_mul.h
    scalar_maximum.h
    scalar_minimum.h
    )

set(generated_files)

foreach(header ${ascendc_api_extend_src})
    get_filename_component(name_part ${header} NAME_WLE)
    set(str_header "${name_part}_str.h")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND cat ${header}
                | sed "'1i\\R\"===('"
                | sed "'$$a\\)===\"'"
                > ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
        DEPENDS ${header}
    )
    list(APPEND generated_files ${CMAKE_CURRENT_BINARY_DIR}/${str_header})
endforeach()

add_custom_target(generated_files ALL DEPENDS ${generated_files})

add_library(ascendc_api_extend INTERFACE)
target_include_directories(ascendc_api_extend INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(ascendc_api_extend generated_files)
