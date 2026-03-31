file(GLOB_RECURSE JIT_EXE_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/*.cc")
message("JIT_EXE_SRC = ${JIT_EXE_SRC}")
add_library(jit_exe STATIC ${JIT_EXE_SRC})

target_include_directories(jit_exe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/pkg_inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/external/acl/error_codes
        ${TOP_DIR}/runtime/pkg_inc/dump
)

target_compile_definitions(jit_exe PRIVATE
        google=ascend_private
        )

target_compile_options(jit_exe PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        -Wno-error=unused-parameter # TODO remove
        -Wno-error=ignored-qualifiers # TODO remove
        -fPIC
        -Dgoogle=ascend_private
)
target_link_options(jit_exe PRIVATE ${AIR_COMMON_LINK_OPTION})
target_link_libraries(jit_exe PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        slog_headers
        metadef_headers
        msprof_headers
        error_manager
        cce_headers
        c_sec
        ascend_protobuf
        runtime
        unified_dlog
        json
        static_mmpa
        ge_compiler
)

install(TARGETS jit_exe OPTIONAL
   ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR})