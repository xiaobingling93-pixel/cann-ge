# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(EXECUTOR_SRC_LIST
    executor/ge_executor.cc
    graph/execute/graph_executor.cc
    graph/execute/model_executor.cc
    graph/load/graph_loader.cc
    graph/load/model_manager/model_manager.cc
    common/profiling/ge_profiling.cc
    common/profiling/command_handle.cc
    common/profiling/profiling_init.cc
    common/dump/dump_utils.cc
    common/dump/dump_callback.cc
)

set(DAVINCI_SRC_LIST
    common/runtime/model_rt_var_manager.cc
    common/runtime/rt_session.cc
    common/dump/opdebug_register.cc
    common/dump/data_dumper.cc
    common/runtime_api_wrapper.cc
    graph/load/model_manager/aipp_utils.cc
    graph/manager/caching_allocator.cc
    graph/manager/graph_mem_allocator.cc
    graph/manager/active_memory_allocator.cc
    graph/manager/mem_manager.cc
    graph/manager/host_mem_allocator.cc
    graph/manager/host_mem_manager.cc
    graph/manager/memory_api.cc
    graph/manager/rdma_pool_allocator.cc
    graph/manager/session_scope_mem_allocator.cc
    common/profiling/profiling_manager.cc
    opskernel_executor/ops_kernel_executor_manager.cc
    graph/manager/util/hcom_ome_util.cc
    graph/manager/trans_var_data_utils.cc
    hybrid/node_executor/aicpu/aicpu_ext_info_handler.cc
    graph/load/model_manager/cpu_queue_schedule.cc
    graph/load/model_manager/davinci_model.cc
    graph/load/model_manager/memory_block_manager.cc
    graph/load/model_manager/reusable_stream_allocator.cc
    graph/load/model_manager/memory_app_type_classifier.cc
    graph/load/model_manager/task_args_refresh_type_classifier.cc
    graph/load/model_manager/task_node_map.cc
    graph/load/model_manager/model_args_layout_planner.cc
    graph/load/model_manager/model_args_manager.cc
    graph/load/model_manager/aicpu_resources.cc
    graph/load/model_manager/model_utils.cc
    graph/load/model_manager/task_info/args_format/args_format_utils.cc
    graph/load/model_manager/kernel/kernel_register_info_builder.cc
    graph/load/model_manager/kernel/model_kernel_handles_manager.cc
    graph/load/model_manager/task_info/fe/cmo_barrier_task_info.cc
    graph/load/model_manager/task_info/fe/cmo_task_info.cc
    graph/load/model_manager/task_info/rts/cmo_addr_task_info.cc
    graph/load/model_manager/task_info/fe/dsa_task_info.cc
    graph/load/model_manager/task_info/dvpp/dvpp_task_info.cc
    graph/load/model_manager/task_info/rts/npu_get_float_status_task_info.cc
    graph/load/model_manager/task_info/rts/npu_clear_float_status_task_info.cc
    graph/load/model_manager/task_info/rts/end_graph_task_info.cc
    graph/load/model_manager/task_info/rts/event_record_task_info.cc
    graph/load/model_manager/task_info/rts/event_wait_task_info.cc
    graph/load/model_manager/task_info/rts/event_mem_record_task_info.cc
    graph/load/model_manager/task_info/rts/event_mem_wait_task_info.cc
    graph/load/model_manager/task_info/rts/notify_record_task_info.cc
    graph/load/model_manager/task_info/rts/notify_wait_task_info.cc
    graph/load/model_manager/task_info/ffts_plus/ffts_plus_context_adapter.cc
    graph/load/model_manager/task_info/ffts_plus/ffts_plus_task_info.cc
    graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.cc
    graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.cc
    graph/load/model_manager/task_info/fe/fusion_start_task_info.cc
    graph/load/model_manager/task_info/fe/fusion_stop_task_info.cc
    graph/load/model_manager/task_info/fe/fusion_task_info.cc
    graph/load/model_manager/task_info/hccl/hccl_task_info.cc
    graph/load/model_manager/task_info/hccl/hccl_util.cc
    graph/load/model_manager/task_info/aicpu/kernel_ex_task_info.cc
    graph/load/model_manager/task_info/fe/kernel_task_info.cc
    graph/load/model_manager/task_info/fe/super_kernel_task_info.cc
    graph/load/model_manager/task_info/fe/update_pc_task_info.cc
    graph/load/model_manager/task_info/rts/nop_task_info.cc
    graph/load/model_manager/task_info/rts/label_goto_ex_task_info.cc
    graph/load/model_manager/task_info/rts/label_set_task_info.cc
    graph/load/model_manager/task_info/rts/label_switch_by_index_task_info.cc
    graph/load/model_manager/task_info/rts/memcpy_addr_async_task_info.cc
    graph/load/model_manager/task_info/rts/memcpy_async_task_info.cc
    graph/load/model_manager/task_info/ge/profiler_trace_task_info.cc
    graph/load/model_manager/task_info/rts/stream_active_task_info.cc
    graph/load/model_manager/task_info/rts/stream_switch_task_info.cc
    graph/load/model_manager/task_info/task_info.cc
    graph/load/model_manager/task_info/task_info_factory.cc
    graph/load/model_manager/task_info/args_io_addrs_updater.cc
    graph/load/model_manager/tbe_kernel_handle.cc
    graph/load/model_manager/zero_copy_offset.cc
)


##################################################################
set(SINGLE_OP_SRC_LIST
    single_op/single_op.cc
    single_op/single_op_manager.cc
    single_op/single_op_model.cc
    single_op/stream_resource.cc
    single_op/task/aicpu_c_c_task_builder.cc
    single_op/task/aicpu_task_builder.cc
    single_op/task/build_task_utils.cc
    single_op/task/op_task.cc
    single_op/task/rts_kernel_task_builder.cc
    single_op/task/tbe_task_builder.cc
    single_op/task/dsa_task_builder.cc
)


##################################################################
set(HYBRID_SRC_LIST
    common/utils/executor_utils.cc
    hybrid/common/npu_memory_allocator.cc
    hybrid/common/tensor_value.cc
    hybrid/common/bin_cache/one_node_single_bin_selector.cc
    hybrid/executor/hybrid_execution_context.cc
    hybrid/executor/hybrid_model_async_executor.cc
    hybrid/executor/hybrid_model_executor.cc
    hybrid/executor/hybrid_model_pipeline_executor.cc
    hybrid/executor/hybrid_model_rt_v2_executor.cc
    hybrid/executor/runtime_v2/rt_v2_executor_factory.cc
    hybrid/executor/runtime_v2/rt_v2_pipeline_executor.cc
    hybrid/executor/runtime_v2/rt_v2_simple_executor.cc
    hybrid/executor/runtime_v2/rt_v2_stage_state.cc
    hybrid/executor/runtime_v2/rt_v2_utils.cc
    hybrid/executor/runtime_v2/scalable_allocator_manager.cc
    hybrid/executor/hybrid_model_rt_v1_executor.cc
    hybrid/executor/hybrid_profiler.cc
    hybrid/executor/node_done_manager.cc
    hybrid/executor/node_state.cc
    hybrid/executor/rt_callback_manager.cc
    hybrid/executor/host_cpu_callback_manager.cc
    hybrid/executor/subgraph_context.cc
    hybrid/executor/subgraph_executor.cc
    hybrid/executor/hybrid_data_flow.cc
    hybrid/executor/resource_manager.cc
    hybrid/executor/worker/execution_engine.cc
    hybrid/executor/worker/shape_inference_engine.cc
    hybrid/executor/cann_tracing_profiler.cc
    hybrid/hybrid_davinci_model.cc
    hybrid/model/graph_item.cc
    hybrid/model/hybrid_model.cc
    hybrid/model/hybrid_model_builder.cc
    hybrid/model/node_item.cc
    hybrid/model/infer/graph_stage_cache.cc
    hybrid/model/infer/node_shape_infer.cc
    hybrid/model/infer/node_shape_propagator.cc
    hybrid/model/infer/shape_utils.cc
    hybrid/model/infer/shape_propagator.cc
    hybrid/model/infer/tensor_desc_observer.cc
    hybrid/model/infer/tensor_desc_holder.cc
    hybrid/node_executor/aicore/aicore_node_executor.cc
    hybrid/node_executor/aicore/aicore_op_task.cc
    hybrid/node_executor/aicore/aicore_task_builder.cc
    hybrid/node_executor/aicpu/aicpu_node_executor.cc
    hybrid/node_executor/compiledsubgraph/known_node_executor.cc
    hybrid/node_executor/controlop/control_op_executor.cc
    hybrid/node_executor/ge_local/data_flow_kernels.cc
    hybrid/node_executor/ge_local/ge_local_node_executor.cc
    hybrid/node_executor/hccl/hccl_node_executor.cc
    hybrid/node_executor/host_cpu/host_cpu_node_executor.cc
    hybrid/node_executor/node_executor.cc
    hybrid/node_executor/partitioned_call/partitioned_call_node_executor.cc
    hybrid/node_executor/rts/rts_node_executor.cc
    hybrid/node_executor/rts/rts_node_task.cc
    hybrid/node_executor/rts/rts_task_factory.cc
    hybrid/node_executor/task_context.cc
    ${AIR_CODE_DIR}/compiler/host_kernels/kernel_utils.cc
    ${AIR_CODE_DIR}/compiler/host_kernels/array_ops/rank_kernel.cc
    ${AIR_CODE_DIR}/compiler/host_kernels/array_ops/shape_kernel.cc
    ${AIR_CODE_DIR}/compiler/host_kernels/array_ops/shape_n_kernel.cc
    ${AIR_CODE_DIR}/compiler/host_kernels/array_ops/size_kernel.cc
    ${AIR_CODE_DIR}/compiler/host_kernels/array_ops/gathershapes_kernel.cc
)

message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
######## mini_acllib_win: libge_executor.a ########
add_library(ge_executor STATIC
    ${EXECUTOR_SRC_LIST}
)

add_dependencies(ge_executor
    graphengine_protos
)

target_compile_options(ge_executor PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:${AIR_COMMON_COMPILE_OPTION}>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
    -Os
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -ffunction-sections
    -fdata-sections
)

target_compile_definitions(ge_executor PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(ge_executor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/runtime/v2
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
    ${TOP_DIR}/hcomm/pkg_inc
    ${TOP_DIR}/hcomm/include
)

target_link_libraries(ge_executor PRIVATE
    intf_pub
    cce_headers
    slog_headers
    msprof_headers
    adump_headers
    mmpa_headers
    metadef_headers
    runtime_headers
    hccl_headers
    datagw_headers
    ascend_protobuf_static
    json
    ascend_dump
    c_sec
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

######## libge_executor.so ########
add_library(ge_executor_shared SHARED
    ${EXECUTOR_SRC_LIST}
)

add_dependencies(ge_executor_shared
    graphengine_protos
)

target_compile_options(ge_executor_shared PRIVATE
    -fno-common
    -Werror
    -O2
    -Wextra
    -Wfloat-equal
    -fvisibility=default
)

target_compile_definitions(ge_executor_shared PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(ge_executor_shared PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
    ${TOP_DIR}/hcomm/pkg_inc
    ${TOP_DIR}/hcomm/include
)

target_link_options(ge_executor_shared PRIVATE
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_libraries(ge_executor_shared
    PRIVATE
        intf_pub
        $<$<TARGET_EXISTS:cce_headers>:cce_headers>
        $<$<TARGET_EXISTS:aicpu_headers>:aicpu_headers>
        msprof_headers
        adump_headers
        mmpa_headers
        hccl_headers
        $<$<TARGET_EXISTS:datagw_headers>:datagw_headers>
        -Wl,--no-as-needed
        ge_common
        ge_common_base
        runtime
        ascend_dump
        davinci_executor
        hybrid_executor
        unified_dlog
        platform
        graph_base
        register
        error_manager
        ascend_protobuf
        c_sec
        json
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        air_headers
)

set_target_properties(ge_executor_shared PROPERTIES
    OUTPUT_NAME ge_executor
)


######## libdavinci_executor.so ########
add_library(davinci_executor SHARED
    ${DAVINCI_SRC_LIST})

add_dependencies(davinci_executor
    graphengine_protos
)

target_compile_options(davinci_executor PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})

target_compile_definitions(davinci_executor PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(davinci_executor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/runtime/v1
    ${AIR_CODE_DIR}/runtime/v2
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
    ${TOP_DIR}/hcomm/pkg_inc
    ${TOP_DIR}/hcomm/include
)

target_link_options(davinci_executor PRIVATE
    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
)

target_link_libraries(davinci_executor
  PRIVATE
    intf_pub
    $<$<TARGET_EXISTS:cce_headers>:cce_headers>
    $<$<TARGET_EXISTS:aicpu_headers>:aicpu_headers>
    msprof_headers
    adump_headers
    mmpa_headers
    hccl_headers
    $<$<TARGET_EXISTS:datagw_headers>:datagw_headers>
    -Wl,--no-as-needed
    graph_base
    ge_common
    ge_common_base
    ascend_dump
    opp_registry
    runtime
    error_manager
    ascend_protobuf
    c_sec
    unified_dlog
    json
    -Wl,--as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
  PUBLIC
    air_headers
)

######## libdavinci_executor.a ########
if ("${PRODUCT}" STREQUAL "ascend031")
    target_clone(davinci_executor davinci_executor_static STATIC)

    add_dependencies(davinci_executor_static graphengine_protos)

    target_compile_options(davinci_executor_static PRIVATE
        -Os
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -ffunction-sections
        -fdata-sections
    )

    set_target_properties(davinci_executor_static PROPERTIES
        OUTPUT_NAME davinci_executor
    )
endif ()


######## libhybrid_executor.so ########
add_library(hybrid_executor SHARED
    ${SINGLE_OP_SRC_LIST}
    ${HYBRID_SRC_LIST}
)

add_dependencies(hybrid_executor
    graphengine_protos
)

target_compile_options(hybrid_executor PRIVATE
    -fno-common -Werror -Wextra -Wfloat-equal -fvisibility=default
    -O2
)

target_compile_definitions(hybrid_executor PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(hybrid_executor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
    ${TOP_DIR}/hcomm/pkg_inc
    ${TOP_DIR}/hcomm/include
)

target_link_options(hybrid_executor PRIVATE
    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
)

target_link_libraries(hybrid_executor
  PRIVATE
    intf_pub
    $<$<TARGET_EXISTS:cce_headers>:cce_headers>
    $<$<TARGET_EXISTS:aicpu_headers>:aicpu_headers>
    msprof_headers
    mmpa_headers
    adump_headers
    $<$<TARGET_EXISTS:datagw_headers>:datagw_headers>
    hccl_headers
    -Wl,--no-as-needed
    graph
    register
    opp_registry
    ge_common
    ge_common_base
    runtime
    davinci_executor
    ascend_dump
    error_manager
    ascend_protobuf
    unified_dlog
    platform
    json
    c_sec
    -Wl,--as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
  PUBLIC
    air_headers
)


####################################################
set(STUB_HEADER_LIST
    ${AIR_CODE_DIR}/runtime/v1/common/utils/executor_utils.h
    ${AIR_CODE_DIR}/runtime/v1/hybrid/common/npu_memory_allocator.h
    ${AIR_CODE_DIR}/runtime/v1/hybrid/hybrid_davinci_model.h
    ${AIR_CODE_DIR}/runtime/v1/single_op/single_op.h
    ${AIR_CODE_DIR}/runtime/v1/single_op/single_op_manager.h
    ${AIR_CODE_DIR}/runtime/v1/single_op/stream_resource.h
)

list(TRANSFORM STUB_HEADER_LIST
    REPLACE "^.*/([^/]+)\\.h$" "${CMAKE_CURRENT_BINARY_DIR}/stub_\\1.cc"
    OUTPUT_VARIABLE STUB_SRC_LIST
)

if(NOT BUILD_PKG_COMPONENT)
    # TODO:待整包下线后删除
    add_custom_command(
        OUTPUT ${STUB_SRC_LIST}
        COMMAND echo "Generating hybrid_executor_stub files."
                && ${HI_PYTHON} ${METADEF_DIR}/tests/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
                && echo "Generating stub files end."
    )
else()
    add_custom_command(
        OUTPUT ${STUB_SRC_LIST}
        COMMAND echo "Generating hybrid_executor_stub files."
                && ${HI_PYTHON} ${GE_METADEF_DIR}/../tests/graph_metadef/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
                && echo "Generating stub files end."
    )
endif()

add_custom_target(stub_hybrid_executor DEPENDS ${STUB_SRC_LIST})

############ stub/libhybrid_executor.so ############
add_library(hybrid_executor_stub SHARED
   ${STUB_SRC_LIST}
)

add_dependencies(hybrid_executor_stub
    graphengine_protos
    stub_hybrid_executor
)

target_compile_options(hybrid_executor_stub PRIVATE
    -fno-common -Wfloat-equal -fvisibility=default -Os
    -Werror=return-type
)

target_compile_definitions(hybrid_executor_stub PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_include_directories(hybrid_executor_stub PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/compiler
    ${CMAKE_CURRENT_LIST_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_link_libraries(hybrid_executor_stub PRIVATE
    intf_pub
    slog_headers
    msprof_headers
    mmpa_headers
    runtime_headers
    metadef_headers
    $<$<TARGET_EXISTS:cce_headers>:cce_headers>
    $<$<TARGET_EXISTS:aicpu_headers>:aicpu_headers>
    c_sec_headers
    ascend_protobuf_shared_headers
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
  PUBLIC
    air_headers
)

set_target_properties(hybrid_executor_stub PROPERTIES
    OUTPUT_NAME hybrid_executor
    LIBRARY_OUTPUT_DIRECTORY stub
)

############ stub/libhybrid_executor.so ############
if ("${PRODUCT}" STREQUAL "ascend031")
    target_clone(hybrid_executor_stub hybrid_executor_static_stub STATIC)

    add_dependencies(hybrid_executor_static_stub
        graphengine_protos
        stub_hybrid_executor
    )

    target_compile_options(hybrid_executor_static_stub PRIVATE
        -ffunction-sections
        -fdata-sections
    )

    set_target_properties(hybrid_executor_static_stub PROPERTIES
        OUTPUT_NAME hybrid_executor
        ARCHIVE_OUTPUT_DIRECTORY stub
    )
endif ()


install(TARGETS hybrid_executor_stub OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)
