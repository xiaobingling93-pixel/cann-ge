#------------------------- public ----------------------------
protobuf_generate(rts_engine_proto rts_engine_proto_c rts_engine_proto_h
                ${METADEF_PROTO_DIR}/task.proto
                ${METADEF_PROTO_DIR}/om.proto
                TARGET)

set(CCEC_PATH ${ASCEND_INSTALL_PATH}/tools/ccec_compiler/bin)
set(TILING_DEPENDS "")
set(INCLUDE_PATHS
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl/basic_api
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl/simt_api
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl/micro_api
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/basic_api
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/simit_api
    -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/micro_api
    -I${ASCEND_INSTALL_PATH}/include/
)

set(rts_engine_source
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/common/util.cpp
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/common/rt_log.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/common/error_code/error_code.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/engine/rts_engine.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ops_kernel_info.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ffts_plus_ops_kernel_info.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/op_factory.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/op_ffts_plus_factory.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_control_op/label_goto_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_control_op/label_goto_ex_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_control_op/label_set_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_control_op/label_switch_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_control_op/label_switch_by_index_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_control_op/stream_switch_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_control_op/stream_switchN_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_control_op/stream_merge_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_control_op/stream_active_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/communication_op/send_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/communication_op/send_notify_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/communication_op/recv_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/communication_op/recv_notify_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/memory_op/memcpy_async_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/memory_op/memcpy_addr_async_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/memory_op/cmo_addr_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/model_op/end_graph_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/model_op/model_exit_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_op/npu_get_float_status_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_op/npu_clear_float_status_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_op/npu_get_float_debug_status_op.cpp
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_op/npu_clear_float_debug_status_op.cpp
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/graph_optimizer/rts_graph_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/graph_optimizer/rts_ffts_plus_graph_optimizer.cc
    ${rts_engine_proto_h}
)



set(rts_engine_include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_BINARY_DIR}/proto/rts_engine_proto
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op
)

set(rts_kernel_builder_src
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ops_kernel_builder.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ffts_plus_ops_kernel_builder.cc
)

#------------------------- rts_engine -------------------------
add_library(rts_engine SHARED
        ${rts_engine_source}
        ${rts_kernel_builder_src}
        ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/op.cc
        )

add_dependencies(rts_engine rts_engine_proto)

target_include_directories(rts_engine PRIVATE
        ${rts_engine_include}
        ${CMAKE_BINARY_DIR}/proto/rts_engine_proto/
        )

target_compile_options(rts_engine PRIVATE
    -Wno-deprecated-declarations
    -std=c++11
    -fno-common
    -fno-strict-aliasing
    -Werror
    -Wextra
    -Wfloat-equal
)

target_compile_definitions(rts_engine PRIVATE
    google=ascend_private
)

target_link_options(rts_engine PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_libraries(rts_engine PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    air_headers
    -Wl,--no-as-needed
    slog
    error_manager
    graph
    register
    platform
    ascend_protobuf
    c_sec
    mmpa
    -Wl,--as-needed
    -lrt
    -ldl
    json
)

install(TARGETS rts_engine OPTIONAL
    EXPORT rts_engine-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

#------------------------- rts_engine_atc -------------------------
add_library(rts_engine_atc SHARED
            ${rts_kernel_builder_src}
            ${rts_engine_source}
            ${rts_engine_proto_h}
            )

add_dependencies(rts_engine_atc rts_engine_proto)

target_include_directories(rts_engine_atc PRIVATE
                    ${rts_engine_include}
                    ${CMAKE_BINARY_DIR}/proto/rts_engine_proto/
                    )

target_compile_definitions(rts_engine_atc PRIVATE
    LOG_CPP
    google=ascend_private
    )

target_compile_options(rts_engine_atc PRIVATE
        -fno-common
        -fno-strict-aliasing
        -Werror
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>>:-Wextra>
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>>:-Wno-error=unused-parameter>
        )

target_link_libraries(rts_engine_atc PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:tsch_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:cce_headers>
    $<BUILD_INTERFACE:atrace_headers>
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    alog
    graph
    register
    runtime
    -Wl,--as-needed
)

set_target_properties(rts_engine_atc PROPERTIES
    OUTPUT_NAME rts_engine
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/atclib
)

install(TARGETS rts_engine_atc
          LIBRARY DESTINATION lib/atclib OPTIONAL
        )

############ librts_kernel_builder.a ############
add_library(rts_kernel_builder_host_static STATIC
        ${rts_kernel_builder_src}
        ${rts_engine_proto_h}
        )
add_dependencies(rts_kernel_builder_host_static rts_engine_proto)

target_include_directories(rts_kernel_builder_host_static PRIVATE
        ${rts_engine_include}
        ${CMAKE_BINARY_DIR}/proto/rts_engine_proto/
        )

target_compile_options(rts_kernel_builder_host_static PRIVATE
        -fno-common
        -fno-strict-aliasing
        -Werror
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>>:-Wextra>
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>>:-Wno-error=unused-parameter>
        )

target_link_libraries(rts_kernel_builder_host_static PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:tsch_headers>
        $<BUILD_INTERFACE:runtime_headers> $<BUILD_INTERFACE:cce_headers> graph
        register
        ascend_protobuf
        c_sec
        alog
        )
target_compile_definitions(rts_kernel_builder_host_static PRIVATE
    LOG_CPP
    google=ascend_private
    )

set(SWITCH_BY_INDEX_SOURCE ${CMAKE_CURRENT_LIST_DIR}/op/switch_by_index.cce)
set(SWITCH_BY_INDEX_OBJ    ${CMAKE_CURRENT_BINARY_DIR}/switch_by_index.o)

add_custom_target(switch_by_index
    DEPENDS ${SWITCH_BY_INDEX_OBJ}
    )

add_custom_command(OUTPUT ${SWITCH_BY_INDEX_OBJ}
    COMMAND ${CCEC_PATH}/ccec ${SWITCH_BY_INDEX_SOURCE} 
    --cce-aicore-only --cce-aicore-arch=dav-n350 -mllvm -cce-aicore-vec-misched=0 
    -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=true -c -o 
    ${SWITCH_BY_INDEX_OBJ} 
    ${INCLUDE_PATHS}
    )

install(FILES ${SWITCH_BY_INDEX_OBJ}
    DESTINATION lib64
    OPTIONAL
    )