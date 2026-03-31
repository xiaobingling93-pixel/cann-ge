file(GLOB_RECURSE META_SRC CONFIGURE_DEPENDS "*.cpp")
add_library(ascir SHARED
        ${META_SRC}
)

target_include_directories(ascir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ascir PRIVATE
        ${ASCEND_3RD_LIB_PATH}/json/include
)
target_compile_definitions(ascir PRIVATE
        google=ascend_private
)

target_compile_options(ascir PRIVATE 
    -Wfloat-equal -Wextra -Wall
)

target_link_libraries(ascir
    PUBLIC
        graph graph_base
        ascir_ops_headers
        c_sec
        unified_dlog
        mmpa
        aihac_ir
        ascend_protobuf
        metadef_headers
    PRIVATE
        $<BUILD_INTERFACE:intf_pub>
)

################################################################################
target_sources(aihac_codegen PRIVATE ${META_SRC}
)

target_include_directories(aihac_codegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_3RD_LIB_PATH}/json/include
)
