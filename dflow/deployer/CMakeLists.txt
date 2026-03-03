# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT DEFINED PRODUCT_SIDE)
    set(PRODUCT_SIDE "host")
endif ()

set(GRPC_PROTO_LIST
        "${AIR_CODE_DIR}/dflow/deployer/proto/deployer.proto"
        )

set(DEPLOYER_PROTO_LIST
        "${AIR_CODE_DIR}/dflow/deployer/proto/deployer.proto"
        )
set(GRPC_SERVER_PROTO_LIST
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/var_manager.proto"
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/flow_model.proto"
        "${AIR_CODE_DIR}/dflow/deployer/proto/deployer.proto"
        )

protobuf_generate(deployer_proto DEPLOYER_PROTO_SRCS DEPLOYER_PROTO_HDRS ${DEPLOYER_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}" TARGET)
protobuf_generate_grpc(deployer GRPC_PROTO_SRCS GRPC_PROTO_HDRS  ${GRPC_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")
protobuf_generate_grpc(deployer GRPC_SERVER_PROTO_SRCS GRPC_SERVER_PROTO_HDRS  ${GRPC_SERVER_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

########################################## libnpu_sched_model_loader #################################
set(NPU_SCHED_MODEL_LOADER_SRC_LIST
        executor/sched_task_info.cc
        executor/npu_sched_model.cc
        executor/npu_sched_model_loader.cc
        executor/npu_sched_model_configurator.cc)


add_library(npu_sched_model_loader SHARED
        ${NPU_SCHED_MODEL_LOADER_SRC_LIST}
        )

target_include_directories(npu_sched_model_loader PRIVATE
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/external/acl/error_codes
        .
        )

target_compile_options(npu_sched_model_loader PRIVATE
        -O2
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -Werror
        )

target_link_options(npu_sched_model_loader PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,ALL
        )

target_link_libraries(npu_sched_model_loader PRIVATE
        intf_pub
        msprof_headers
        datagw_headers
        -Wl,--no-as-needed
        unified_dlog
        runtime
        c_sec
        ge_common
        ge_common_base
        error_manager
        dl
        -Wl,--as-needed
        data_flow_base
        )

########################################## libmodel_deployer #################################
set(HELPER_COMMON_SRC_LIST
        common/config/config_parser.cc
        common/config/numa_config_manager.cc
        common/config/configurations.cc
        common/config/device_debug_config.cc
        common/config/json_parser.cc
        common/data_flow/event/proxy_event_manager.cc
        common/data_flow/queue/heterogeneous_exchange_service.cc
        common/message_handle/message_client.cc
        common/message_handle/message_server.cc
        common/data_flow/route/rank_table_builder.cc
        common/mem_grp/memory_group_manager.cc
        common/subprocess/subprocess_manager.cc
        common/utils/heterogeneous_profiler.cc
        common/utils/memory_statistic_manager.cc)


set(HELPER_FLOW_RM_SRC_LIST
        deploy/flowrm/flowgw_client.cc
        deploy/flowrm/flowgw_client_manager.cc
        deploy/flowrm/flow_route_manager.cc
        deploy/flowrm/flow_route_planner.cc
        deploy/flowrm/heterogeneous_exchange_deployer.cc
        deploy/flowrm/network_manager.cc
        deploy/flowrm/tsd_client.cc
        )

set(HELPER_RESOURCE_MANAGER_SRC_LIST
        deploy/resource/deployer_port_distributor.cc
        deploy/resource/device_info.cc
        deploy/resource/heterogeneous_deploy_planner.cc
        deploy/resource/node_info.cc
        deploy/resource/resource_allocator.cc
        deploy/resource/resource_manager.cc
        )

set(HELPER_DEPLOYER_SRC_LIST
        deploy/deployer/deploy_context.cc
        deploy/deployer/deploy_state.cc
        deploy/deployer/deployer_authentication.cc
        deploy/deployer/deployer.cc
        deploy/deployer/deployer_proxy.cc
        deploy/deployer/deployer_service_impl.cc
        deploy/deployer/deployer_var_manager.cc
        deploy/deployer/heterogeneous_model_deployer.cc
        deploy/deployer/master_model_deployer.cc
        deploy/abnormal_status_handler/abnormal_status_handler.cc
        deploy/abnormal_status_handler/device_abnormal_status_handler.cc
        )

set(HELPER_DEPLOYER_RPC_SRC_LIST
        deploy/rpc/deployer_client.cc
        deploy/rpc/deployer_server.cc
        )

set(HELPER_EXEC_FWK_SRC_LIST
        deploy/execfwk/executor_manager.cc
        deploy/execfwk/pne_executor_client.cc
        deploy/execfwk/builtin_executor_client.cc
        deploy/execfwk/udf_proxy_client.cc
        deploy/execfwk/udf_executor_client.cc)

set(LIB_MODEL_DEPLOYER_SRC_LIST
        ${HELPER_COMMON_SRC_LIST}
        ${HELPER_DEPLOYER_SRC_LIST}
        ${HELPER_EXEC_FWK_SRC_LIST}
        ${HELPER_FLOW_RM_SRC_LIST}
        ${HELPER_RESOURCE_MANAGER_SRC_LIST}
        ${HELPER_DEPLOYER_RPC_SRC_LIST}
        deploy/model_send/flow_model_sender.cc
        deploy/model_recv/flow_model_receiver.cc
        deploy/heterogeneous_execution_runtime.cc)

add_library(model_deployer SHARED
        ${LIB_MODEL_DEPLOYER_SRC_LIST}
        ${GRPC_PROTO_HDRS}
        ${GRPC_PROTO_SRCS}
        )

target_include_directories(model_deployer PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/compiler
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto
        ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
        ${CMAKE_BINARY_DIR}/proto_grpc/deployer
        ${METADEF_DIR}/pkg_inc
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/external/acl/error_codes
        .
        )

target_compile_options(model_deployer PRIVATE
        -O2
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -Werror
        )

add_dependencies(model_deployer graphengine_protos data_flow_base_proto)

target_compile_definitions(model_deployer PRIVATE
        google=ascend_private
        )

target_link_options(model_deployer PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,ALL
        )

target_link_libraries(model_deployer PRIVATE
        intf_pub
        ascendcl
        msprof_headers
        cce_headers
        datagw_headers
        metadef_headers
        hccl_headers
        -Xlinker "-("
        $<TARGET_NAME_IF_EXISTS:gRPC::grpc++>
        $<TARGET_NAME_IF_EXISTS:gRPC::grpc>
        -Xlinker "-)"
        -Wl,--no-as-needed
        unified_dlog
        static_mmpa
        runtime
        ascend_protobuf
        c_sec
        ge_common
        ge_common_base
        data_flow_base
        ge_executor_shared
        gert
        error_manager
        dgw_client
        dl
        -Wl,--as-needed
        json
        )

########################################## model_deployer_daemon ###############################
set(DEPLOYER_DAEMON_SRC_FILE
        daemon/daemon_client_manager.cc
        daemon/daemon_service.cc
        daemon/deployer_daemon_client.cc
        daemon/model_deployer_daemon.cc
        daemon/deployer_daemon.cc
        )

set(DEPLOYER_DAEMON_INC_LIST
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/base
        .
        ${CMAKE_BINARY_DIR}/proto_grpc/deployer
        )

add_executable(deployer_daemon
        ${DEPLOYER_DAEMON_SRC_FILE}
        ${GRPC_SERVER_PROTO_HDRS}
        )

target_include_directories(deployer_daemon PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${DEPLOYER_DAEMON_INC_LIST}
        ${METADEF_DIR}/pkg_inc
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/external/acl/error_codes
        )

target_compile_options(deployer_daemon PRIVATE
        -O2
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -Werror
        )

target_compile_definitions(deployer_daemon PRIVATE
        google=ascend_private
        )

target_link_options(deployer_daemon PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(deployer_daemon PRIVATE
        intf_pub
        msprof_headers
        datagw_headers
        air_headers
        -Wl,--no-as-needed
        unified_dlog
        alog
        static_mmpa
        c_sec
        ascend_protobuf
        error_manager
        rt
        dl
        runtime
        dgw_client
        ge_common_base
        data_flow_base
        model_deployer
        ge_executor_shared
        -Wl,--as-needed
        json
        )

########################################## npu_executor ###############################
set(EXECUTOR_COMMON_SRC_LIST
        common/config/json_parser.cc
        common/utils/heterogeneous_profiler.cc
        common/utils/memory_statistic_manager.cc
        common/data_flow/queue/heterogeneous_exchange_service.cc
        common/message_handle/message_client.cc
        common/message_handle/message_server.cc)

set(NPU_EXECUTOR_INC_LIST
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/compiler
        .
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto
        ${CMAKE_BINARY_DIR}/proto/deployer_proto
        )

set(NPU_EXECUTOR_SRC_LIST
        executor/cpu_id_resource_manager.cc
        executor/cpu_sched_event_dispatcher.cc
        executor/cpu_sched_model.cc
        executor/cpu_sched_model_builder.cc
        executor/cpu_tasks.cc
        executor/dynamic_model_executor.cc
        executor/engine_daemon.cc
        executor/event_handler.cc
        executor/executor_context.cc
        executor/npu_executor.cc
        executor/proxy_dynamic_model_executor.cc
        )

add_executable(npu_executor_main
        ${NPU_EXECUTOR_SRC_LIST}
        ${EXECUTOR_COMMON_SRC_LIST}
        ${DEPLOYER_PROTO_HDRS}
        ${DEPLOYER_PROTO_SRCS}
        )

add_dependencies(npu_executor_main graphengine_protos deployer_proto)

target_include_directories(npu_executor_main PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${NPU_EXECUTOR_INC_LIST}
        ${METADEF_DIR}/pkg_inc
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/external/acl/error_codes
        )

target_compile_options(npu_executor_main PRIVATE
        -O2
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -Werror
        )

target_compile_definitions(npu_executor_main PRIVATE
        google=ascend_private)

target_link_options(npu_executor_main PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,ALL)

target_link_libraries(npu_executor_main PRIVATE
        intf_pub
        datagw_headers
        cce_headers
        hccl_headers
        air_headers
        -Wl,--no-as-needed
        ascend_protobuf
        unified_dlog
        alog
        error_manager
        c_sec
        runtime
        profapi_share
        ascend_hal_stub
        msprofiler_fwk_share
        adump_server
        $<$<BOOL:${ENABLE_OPEN_SRC}>:adcore>
        metadef
        static_mmpa
        json
        dl
        ge_common_base
        rt
        ge_common
        graph
        graph_base
        davinci_executor
        hybrid_executor
        gert
        npu_sched_model_loader
        ge_executor_shared
        -Wl,--as-needed
        data_flow_base
        )

if ("${BUILD_COMPONENT}" STREQUAL "dflow-executor")
        target_link_libraries(npu_executor_main PRIVATE
                -Wl,--no-as-needed
                acl_mdl
                acl_rt)
else ()
        target_link_libraries(npu_executor_main PRIVATE
                -Wl,--no-as-needed
                ascendcl)
endif ()


########################################## host_cpu_executor ###############################
set(HOST_CPU_EXECUTOR_INC_LIST
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/compiler
        .
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto
        ${CMAKE_BINARY_DIR}/proto/deployer_proto
        )

set(HOST_CPU_EXECUTOR_SRC_LIST
        executor/cpu_id_resource_manager.cc
        executor/cpu_sched_event_dispatcher.cc
        executor/cpu_sched_model.cc
        executor/cpu_sched_model_builder.cc
        executor/cpu_tasks.cc
        executor/dynamic_model_executor.cc
        executor/engine_daemon.cc
        executor/event_handler.cc
        executor/executor_context.cc
        executor/host_cpu_executor.cc
        executor/proxy_dynamic_model_executor.cc
        )

add_executable(host_cpu_executor_main
        ${HOST_CPU_EXECUTOR_SRC_LIST}
        ${EXECUTOR_COMMON_SRC_LIST}
        ${DEPLOYER_PROTO_HDRS}
        ${DEPLOYER_PROTO_SRCS}
        )

add_dependencies(host_cpu_executor_main graphengine_protos deployer_proto)

target_include_directories(host_cpu_executor_main PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${HOST_CPU_EXECUTOR_INC_LIST}
        ${METADEF_DIR}/pkg_inc
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/external/acl/error_codes
        )

target_compile_options(host_cpu_executor_main PRIVATE
        -O2
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -Werror
        )

target_compile_definitions(host_cpu_executor_main PRIVATE
        google=ascend_private)

target_link_options(host_cpu_executor_main PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,ALL)

target_link_libraries(host_cpu_executor_main PRIVATE
        intf_pub
        datagw_headers
        hccl_headers
        air_headers
        cce_headers
        -Wl,--no-as-needed
        ascend_protobuf
        unified_dlog
        alog
        error_manager
        c_sec
        runtime
        ascend_hal_stub
        msprofiler_fwk_share
        profapi_share
        adump_server
        $<$<BOOL:${ENABLE_OPEN_SRC}>:adcore>
        metadef
        static_mmpa
        json
        dl
        rt
        ge_common_base
        ge_common
        graph
        graph_base
        davinci_executor
        hybrid_executor
        gert
        npu_sched_model_loader
        ge_executor_shared
        -Wl,--as-needed
        data_flow_base
        )

if ("${BUILD_COMPONENT}" STREQUAL "dflow-executor")
        target_link_libraries(host_cpu_executor_main PRIVATE
                -Wl,--no-as-needed
                acl_rt
                acl_mdl
                )
else ()
        target_link_libraries(host_cpu_executor_main PRIVATE
                -Wl,--no-as-needed
                ascendcl)
endif ()

install(TARGETS npu_sched_model_loader model_deployer deployer_daemon npu_executor_main host_cpu_executor_main OPTIONAL
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
        RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}
        )

add_custom_target(
        monitor.sh ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/monitor.sh
)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/monitor.sh
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/monitor_cpu.sh ${CMAKE_CURRENT_BINARY_DIR}/monitor.sh
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monitor.sh OPTIONAL
        DESTINATION ${INSTALL_RUNTIME_DIR}
)
