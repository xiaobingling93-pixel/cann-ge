add_subdirectory(generator)
######### libaihac_ir.so #############
file(GLOB_RECURSE UTILS_SRCS CONFIGURE_DEPENDS "utils/*.cc")
add_library(aihac_ir SHARED
    core/ascendc_ir.cc
    ${UTILS_SRCS}
)
target_compile_options(aihac_ir PRIVATE -DNO_METADEF_ABI_COMPATIABLE)

target_compile_options(aihac_ir PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(aihac_ir PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(aihac_ir PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_link_options(aihac_ir PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(aihac_ir
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        c_sec
        slog
        json
        platform
        -Wl,--as-needed
        ascend_protobuf_shared_headers
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        graph
        graph_base
        error_manager
        aihac_symbolizer
    PUBLIC
        metadef_headers
)