# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ---- Test coverage ----

if (NOT DEFINED PRODUCT_SIDE)
    set(PRODUCT_SIDE "host")
endif ()

set(HELPER_RUNTIME_SRC
    ${AIR_CODE_DIR}/dflow/deployer/common/config/config_parser.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/config/numa_config_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/config/configurations.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/config/device_debug_config.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/config/json_parser.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/data_flow/queue/heterogeneous_exchange_service.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/message_handle/message_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/message_handle/message_server.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/data_flow/route/rank_table_builder.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/data_flow/event/proxy_event_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/mem_grp/memory_group_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/subprocess/subprocess_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/utils/heterogeneous_profiler.cc
    ${AIR_CODE_DIR}/dflow/deployer/common/utils/memory_statistic_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/daemon/daemon_client_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/daemon/daemon_service.cc
    ${AIR_CODE_DIR}/dflow/deployer/daemon/deployer_daemon_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/daemon/model_deployer_daemon.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/heterogeneous_execution_runtime.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deploy_context.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deploy_state.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_authentication.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_proxy.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_service_impl.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_var_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/heterogeneous_model_deployer.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/master_model_deployer.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/abnormal_status_handler/abnormal_status_handler.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/abnormal_status_handler/device_abnormal_status_handler.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/executor_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/builtin_executor_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/pne_executor_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/udf_executor_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/udf_proxy_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flowgw_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flowgw_client_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flow_route_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flow_route_planner.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/heterogeneous_exchange_deployer.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/network_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/tsd_client.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/model_recv/flow_model_receiver.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/model_send/flow_model_sender.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/resource/deployer_port_distributor.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/resource/device_info.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/resource/node_info.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/resource/heterogeneous_deploy_planner.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/resource/resource_allocator.cc
    ${AIR_CODE_DIR}/dflow/deployer/deploy/resource/resource_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/cpu_id_resource_manager.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/cpu_sched_event_dispatcher.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/cpu_sched_model.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/cpu_sched_model_builder.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/cpu_tasks.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/dynamic_model_executor.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/event_handler.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/executor_context.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/engine_daemon.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/engine_thread.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/sched_task_info.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/npu_sched_model_loader.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/npu_sched_model.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/npu_sched_model_configurator.cc
    ${AIR_CODE_DIR}/dflow/deployer/executor/proxy_dynamic_model_executor.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/dgw_client_st_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/tsd_client_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/caas_dataflow_auth_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/re_mem_group_st_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/process_utils_stub.cc
)

set(GRPC_RELATED_SRC
        ${AIR_CODE_DIR}/dflow/deployer/deploy/rpc/deployer_client.cc
        ${AIR_CODE_DIR}/dflow/deployer/deploy/rpc/deployer_server.cc
        )

set(GRPC_STUB_SRC
        ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/deployer_client_stub.cc
        )

list(APPEND STUB_LIBS
    c_sec
    slog_stub
    runtime_stub
    mmpa_stub
    profiler_stub
    hccl_stub
    error_manager
    ascend_protobuf
    json
)

# ---- Target : helper runtime ----
set(GRPC_PROTO_LIST
    "${AIR_CODE_DIR}/dflow/deployer/proto/deployer.proto"
)

protobuf_generate_grpc(deployer GRPC_PROTO_SRCS GRPC_PROTO_HDRS ${GRPC_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

file(GLOB_RECURSE FLOW_GRAPH_SRCS CONFIGURE_DEPENDS
        ${DFLOW_CODE_DIR}/flow_graph/**.cc
)
add_library(helper_runtime SHARED
    ${HELPER_RUNTIME_SRC}
    ${GRPC_RELATED_SRC}
    ${GRPC_PROTO_SRCS}
    ${GRPC_PROTO_HDRS}
    ${FLOW_GRAPH_SRCS}
)

target_compile_definitions(helper_runtime PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
)

target_include_directories(helper_runtime PUBLIC
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}/proto_grpc/deployer
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/runtime/v1
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/runtime
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/dflow/inc
    ${AIR_CODE_DIR}/dflow/deployer
    ${CMAKE_BINARY_DIR}/proto
    ${CMAKE_BINARY_DIR}/opensrc/json/include
    ${CMAKE_BINARY_DIR}/grpc_build-prefix/src/grpc_build/include
    ${CMAKE_BINARY_DIR}/protoc_grpc_build-prefix/src/protoc_grpc_build/include
    ${CMAKE_BINARY_DIR}/proto/ge
    ${CMAKE_BINARY_DIR}/proto/ge/proto
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto
    ${AIR_CODE_DIR}/dflow
    ${AIR_CODE_DIR}/dflow/inc
    ${AIR_CODE_DIR}/dflow/inc/data_flow
)

target_compile_options(helper_runtime PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(helper_runtime PUBLIC
    intf_pub
    mmpa_headers
    slog_headers
    metadef_headers
    runtime_headers
    cce_headers
    datagw_headers
    ascendcl_headers
    ascend_hal_headers
    adump_headers
    flow_graph_protos_obj
    gRPC::grpc++
    gRPC::grpc
    -Wl,--no-as-needed
    ${STUB_LIBS}
    memset_stubs
    dl
    -Wl,--as-needed
    -lrt -ldl -lpthread -lgcov
    data_flow_base
)

# ---- Target : helper runtime no_grpc ----
set(PROTO_LIST
        "${AIR_CODE_DIR}/runtime/proto/deployer.proto"
        )

protobuf_generate(deployer PROTO_SRCS PROTO_HDRS ${GRPC_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

add_library(helper_runtime_no_grpc SHARED
        ${HELPER_RUNTIME_SRC}
        ${GRPC_STUB_SRC}
        ${PROTO_SRCS}
        ${PROTO_HDRS}
        )

target_compile_definitions(helper_runtime_no_grpc PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
        )

target_include_directories(helper_runtime_no_grpc PUBLIC
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${CMAKE_BINARY_DIR}/proto_grpc/deployer
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/runtime
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/dflow/deployer
        ${CMAKE_BINARY_DIR}/proto
        ${CMAKE_BINARY_DIR}/opensrc/json/include
        ${CMAKE_BINARY_DIR}/grpc_build-prefix/src/grpc_build/include
        ${CMAKE_BINARY_DIR}/protoc_grpc_build-prefix/src/protoc_grpc_build/include
        ${CMAKE_BINARY_DIR}/proto/ge
        ${CMAKE_BINARY_DIR}/proto/ge/proto
        ${CMAKE_BINARY_DIR}/proto/data_flow_protos
        ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
        ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto
        ${AIR_CODE_DIR}/dflow
        ${AIR_CODE_DIR}/dflow/inc
        ${AIR_CODE_DIR}/dflow/inc/data_flow
        )

target_compile_options(helper_runtime_no_grpc PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Werror=format
        )

target_link_libraries(helper_runtime_no_grpc PUBLIC
        intf_pub
        mmpa_headers
        slog_headers
        metadef_headers
        -Wl,--no-as-needed
        ${STUB_LIBS}
        dl
        -Wl,--as-needed
        -lrt -ldl -lpthread -lgcov
        data_flow_base
        )
