include(${GE_METADEF_DIR}/cmake/build_type.cmake)
add_library(aihac_ir_register SHARED
        ascir_register.cc
        ascir_registry.cc
        )
target_clone_compile_and_link_options(graph_base aihac_ir_register)
target_include_directories(aihac_ir_register PRIVATE
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/inc/graph
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/graph/utils
        ${CMAKE_BINARY_DIR}/proto/metadef_protos)
target_link_libraries(aihac_ir_register PRIVATE
        intf_pub
        -Wl,--no-as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        graph
        PUBLIC
        aihac_ir
        ascend_protobuf_shared_headers
        slog
        c_sec
        metadef_headers)

add_library(ascir_generate SHARED
        generator.cc
        )
target_clone_compile_and_link_options(aihac_ir_register ascir_generate)

target_include_directories(ascir_generate PRIVATE
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/inc/graph
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/graph/utils
        ${CMAKE_BINARY_DIR}/proto/metadef_protos)

target_link_libraries(ascir_generate PRIVATE
        intf_pub
        -Wl,--no-as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        aihac_ir_register
        aihac_ir
        ascend_protobuf_shared_headers
        PUBLIC
        metadef_headers)

add_executable(ascir_ops_header_generator ascir_ops_generator_main.cc)
target_compile_options(ascir_ops_header_generator PRIVATE -Wextra -Wfloat-equal)
target_compile_definitions(ascir_ops_header_generator PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)
target_link_libraries(ascir_ops_header_generator PRIVATE ascir_generate
        intf_pub
        -Wl,--no-as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        c_sec
        static_mmpa)
include(generator.cmake)


