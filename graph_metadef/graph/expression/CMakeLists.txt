add_library(aihac_symbolizer SHARED
    expression.cc
    expression_impl.cc
    expr_print_manager.cc
    symbol_operator.cc
    symbolic_utils.cc
    symbol_checker.cc
    expr_parser.cc
    scanner.cc
    attr_group_shape_env_attr.cc
    attr_group_symbolic_desc_attr.cc
    guard_dfx_context.cc
)
target_compile_options(aihac_symbolizer PRIVATE -DNO_METADEF_ABI_COMPATIABLE)

target_compile_options(aihac_symbolizer PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(aihac_symbolizer PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(aihac_symbolizer PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_link_options(aihac_symbolizer PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(aihac_symbolizer
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        c_sec
        slog
        json
        platform
        symengine
        error_manager
        graph_base
        graph
        slog
        Boost::boost
        -Wl,--as-needed
        ascend_protobuf_shared_headers
        ascend_protobuf
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)
set_target_properties(aihac_symbolizer PROPERTIES
        CXX_EXTENSIONS NO)
