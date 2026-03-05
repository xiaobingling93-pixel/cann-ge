# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(PROTO_LIST
    "${METADEF_PROTO_DIR}/om.proto"
    "${METADEF_PROTO_DIR}/ge_ir.proto"
    "${METADEF_PROTO_DIR}/task.proto"
    "${METADEF_PROTO_DIR}/ge_api.proto"
    "${METADEF_PROTO_DIR}/insert_op.proto"
    "${METADEF_PROTO_DIR}/dump_task.proto"
    "${METADEF_PROTO_DIR}/fwk_adapter.proto"
    "${METADEF_PROTO_DIR}/op_mapping.proto"
    "${METADEF_PROTO_DIR}/optimizer_priority.proto"
    "${METADEF_PROTO_DIR}/ge_api.proto"
    "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
    "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
    "${METADEF_PROTO_DIR}/tensorflow/types.proto"
    "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
    "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
    "${METADEF_PROTO_DIR}/var_manager.proto"
    "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
    "${METADEF_PROTO_DIR}/task_mobile.proto"
)

set(DEPLOYER_PROTO_LIST
    "${AIR_CODE_DIR}/dflow/deployer/proto/deployer.proto"
)

set (UDF_PROTO_LIST
    "${AIR_CODE_DIR}/dflow/base/proto/udf_attr.proto"
    "${AIR_CODE_DIR}/dflow/base/proto/udf_def.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")
protobuf_generate_grpc(deployer_proto DEPLOYER_PROTO_SRCS DEPLOYER_PROTO_HDRS ${DEPLOYER_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")
protobuf_generate(ge UDF_PROTO_SRCS UDF_PROTO_HDRS ${UDF_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/data_flow_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)
include_directories(${CMAKE_BINARY_DIR}/proto_grpc/deployer_proto)

include_directories(${AIR_CODE_DIR}/base)
include_directories(${AIR_CODE_DIR}/compiler)
include_directories(${AIR_CODE_DIR}/runtime/v1)
include_directories(${AIR_CODE_DIR}/api/session)
include_directories(${AIR_CODE_DIR}/dflow/compiler/session)
include_directories(${AIR_CODE_DIR}/inc)
include_directories(${AIR_CODE_DIR}/inc/graph_metadef)
include_directories(${AIR_CODE_DIR}/inc/external)
include_directories(${AIR_CODE_DIR}/inc/framework)
include_directories(${AIR_CODE_DIR}/runtime/v2)

include_directories(${PARSER_DIR})
include_directories(${AIR_CODE_DIR}/parser)
include_directories(${AIR_CODE_DIR}/parser/parser)
include_directories(${AIR_CODE_DIR}/inc/parser)
include_directories(${AIR_CODE_DIR}/inc/parser/external)

include_directories(${AIR_CODE_DIR}/tests/)
include_directories(${AIR_CODE_DIR}/tests/dflow/runner/ut/ge)
include_directories(${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/stub)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_graph_dsl/include)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include/)
include_directories(${AIR_CODE_DIR}/tests/framework/easy_graph/include)
include_directories(${AIR_CODE_DIR}/tests/framework/macro_utils)
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/pkg_inc)
include_directories(${METADEF_DIR})
include_directories(${ASCGEN_DIR}/inc)

# test files
set(COMMON_TEST_FILES
    "graph/passes/graph_builder_utils.cc"
    "common/model_saver_unittest.cc"
    "test.cc"
)

set(HETEROGENEOUS_MODEL_TEST_FILES
    "common/execution_runtime_unittest.cc"
    "common/model_relation_unittest.cc"
    "common/model_introduction_unittest.cc"
    "common/deploy_planner_unittest.cc"
    "runtime/heterogeneous/stub/stub_models.cc"
    "graph/load/heterogeneous_model_executor_unittest.cc"
    "graph/load/inner_process_msg_forwarding_unittest.cc"
    "graph/load/data_flow_exception_handler_unittest.cc"
    "graph/load/data_flow_data_aligner_unittest.cc"
    "graph/load/data_flow_info_impl_unittest.cc"
    "graph/load/flow_msg_unittest.cc"
)

set(UDF_TEST_FILES
    "pne/udf/udf_unittest.cc"
)

set(PNE_MODEL_TEST_FILES
    "pne/model/flow_model_builder_unittest.cc"
    "pne/model/flow_model_cache_unittest.cc"
    "pne/model/flow_model_helper_unittest.cc"
)

set(DATA_FLOW_GRAPH_TEST_FILES
    "pne/data_flow_graph/compile_config_json_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_pass_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_model_relation_builder_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_utils_unittest.cc"
    "pne/data_flow_graph/funciton_compile_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_auto_deployer_unittest.cc"
    "pne/data_flow_graph/process_point_loader_unittest.cc"
    "pne/data_flow_graph/inner_pp_loader_unittest.cc"
)

SET(HELPER_RUNTIME_TEST_FILES
    runtime/heterogeneous/common/config/config_parser_unittest.cc
    runtime/heterogeneous/common/config/configurations_unittest.cc
    runtime/heterogeneous/common/config/numa_config_manager_unittest.cpp
    runtime/heterogeneous/common/config/device_debug_config_unittest.cc
    runtime/heterogeneous/common/config/json_parser_unittest.cc
    runtime/heterogeneous/common/data_flow/queue/heterogeneous_exchange_service_unittest.cc
    runtime/heterogeneous/common/data_flow/event/proxy_event_manager_unittest.cc
    runtime/heterogeneous/common/data_flow/route/rank_table_builder_utest.cc
    runtime/heterogeneous/common/mem_grp/memory_group_manager_unittest.cc
    runtime/heterogeneous/common/subprocess/subprocess_manager_unittest.cc
    runtime/heterogeneous/common/utils/memory_statistic_manager_unittest.cc
    runtime/heterogeneous/daemon/client_manager_unittest.cc
    runtime/heterogeneous/daemon/daemon_service_unittest.cc
    runtime/heterogeneous/daemon/model_deployer_daemon_unittest.cc
    runtime/heterogeneous/deploy/abnormal_status_handler/device_abnormal_status_handler_unittest.cc
    runtime/heterogeneous/deploy/deployer/deploy_context_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_proxy_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_authentication_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_service_impl_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_var_manager_unittest.cc
    runtime/heterogeneous/deploy/deployer/master_model_deployer_unittest.cc
    runtime/heterogeneous/deploy/execfwk/builtin_executor_client_unittest.cc
    runtime/heterogeneous/deploy/execfwk/udf_proxy_client_unittest.cc
    runtime/heterogeneous/deploy/execfwk/udf_executor_client_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flowgw_client_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flowgw_client_manager_unittest.cc
    runtime/heterogeneous/deploy/flowrm/tsd_client_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flow_route_manager_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flow_route_planner_unittest.cc
    runtime/heterogeneous/deploy/flowrm/heterogeneous_exchange_deployer_unittest.cc
    runtime/heterogeneous/deploy/model_send/flow_model_sender_unittest.cc
    runtime/heterogeneous/deploy/resource/deployer_port_distributor_unittest.cc
    runtime/heterogeneous/deploy/resource/heterogeneous_deploy_planner_unittest.cc
    runtime/heterogeneous/deploy/resource/resource_manager_unittest.cc
    runtime/heterogeneous/deploy/rpc/deployer_client_unittest.cc
    runtime/heterogeneous/deploy/heterogeneous_execution_runtime_unittest.cc
    runtime/heterogeneous/executor/dynamic_model_executor_unittest.cc
    runtime/heterogeneous/executor/cpu_tasks_unittest.cc
    runtime/heterogeneous/executor/npu_sched_model_unittest.cc
    runtime/heterogeneous/executor/engine_thread_unittest.cc
    runtime/heterogeneous/executor/engine_daemon_unittest.cc
    runtime/heterogeneous/executor/event_handler_unittest.cc
    runtime/heterogeneous/executor/executor_context_unittest.cc
    runtime/heterogeneous/executor/proxy_dynamic_model_executor_unittest.cc
    runtime/heterogeneous/stub/dgw_client_stub.cc
    runtime/heterogeneous/stub/heterogeneous_stub_env.cc
    runtime/heterogeneous/stub/re_mem_group_stub.cc
    runtime/heterogeneous/stub/stub_models.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/caas_dataflow_auth_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/process_utils_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/tsd_client_stub.cc
    ${AIR_CODE_DIR}/tests/dflow/udf/stub/slog_stub.cpp
)

SET(DFLOW_EXECUTOR_TEST_FILES
        "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/executor/flow_model_manager_unittest.cc"
)

SET(DFLOW_SESSION_TEST_FILES
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_api_unittest.cc"
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_graph_manager_unittest.cc"
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_session_manager_unittest.cc"
)



#build runtime
add_library(helper_ut_runtime SHARED ${PROTO_HDRS} ${DEPLOYER_PROTO_HDRS} ${DEPLOYER_PROTO_SRCS})

target_compile_definitions(helper_ut_runtime PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(helper_ut_runtime PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(helper_ut_runtime PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(helper_ut_runtime PUBLIC
    intf_pub
    metadef_headers
    slog_headers
    cce_headers
    msprof_headers
    datagw_headers
    ascend_hal_headers
    hccl_headers
    adump_headers
    ascend_protobuf
    gRPC::grpc++
    gRPC::grpc
    json
    c_sec
    runtime_stub
    mmpa_stub
    -Wl,--no-as-needed
    memset_stubs
    dl
    -Wl,--as-needed
    -lrt -ldl -lpthread -lgcov
    ascendcl_stub
    data_flow_base
)

target_include_directories(helper_ut_runtime PRIVATE
    ${AIR_CODE_DIR}/runtime/device/remote
    ${AIR_CODE_DIR}/runtime/communication
    ${AIR_CODE_DIR}/runtime
    ${AIR_CODE_DIR}/dflow/deployer
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/runtime/deploy/deployer/device_mgr
    ${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/stub
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
    "${CMAKE_BINARY_DIR}/proto/data_flow_protos"
    "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto"
    "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto"
    "${AIR_CODE_DIR}/dflow"
    "${AIR_CODE_DIR}/dflow/inc"
    "${AIR_CODE_DIR}/dflow/inc/data_flow"
)

# helper ut
add_executable(ut_libge_helper_utest
    ${COMMON_TEST_FILES}
    ${HELPER_RUNTIME_TEST_FILES}
    ${UDF_TEST_FILES}
    ${PNE_MODEL_TEST_FILES}
    ${HETEROGENEOUS_MODEL_TEST_FILES}
    ${DFLOW_EXECUTOR_TEST_FILES}
    ${DFLOW_SESSION_TEST_FILES}
    ${DATA_FLOW_GRAPH_TEST_FILES}
    "${AIR_CODE_DIR}/dflow/deployer/executor/engine_thread.cc"
    "${AIR_CODE_DIR}/dflow/deployer/executor/event_handler.cc"
)

target_include_directories(ut_libge_helper_utest PRIVATE
        ${AIR_CODE_DIR}/dflow/deployer/
        ${AIR_CODE_DIR}/runtime
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/tests
        ${AIR_CODE_DIR}/tests/dflow/runner/ut
        ${AIR_CODE_DIR}/tests/framework/ge_running_env/include
        ${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/heterogeneous/
        ${AIR_CODE_DIR}/tests/depends/runtime/src
        ${AIR_CODE_DIR}/tests/depends/ascendcl/src
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${ASCEND_INSTALL_PATH}/include
        ${ASCGEN_DIR}/inc
        "${CMAKE_BINARY_DIR}/proto/data_flow_protos"
        "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto"
        "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto"
        "${AIR_CODE_DIR}/dflow"
        "${AIR_CODE_DIR}/dflow/inc"
        "${AIR_CODE_DIR}/dflow/inc/data_flow"
        )

target_compile_options(ut_libge_helper_utest PRIVATE -g
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -Wno-address
    -Wno-write-strings
    -fno-access-control
)

target_compile_definitions(ut_libge_helper_utest PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_link_options(ut_libge_helper_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_libge_helper_utest
    intf_pub
    metadef_headers
    ascendcl_stub
    datagw_headers
    ascend_hal_headers
    runtime_headers
    flow_func
    dflow_runner
    ge_compiler
    flow_graph
    udf_compiler
    model_deployer
    deployer_daemon_static
    npu_executor_static
    npu_sched_model_loader
    data_flow_base
    ge_runner_v2
    ge_runner
    cpu_compiler
    gert
    -Wl,--whole-archive
    es_ge_test
    helper_ut_runtime ge_runtime_stub
    ge_running_env
    rt2_registry_static
    -Wl,--no-whole-archive
    GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lpthread
    -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
    data_flow_base
)

add_test(NAME ut_libge_helper_utest COMMAND ut_libge_helper_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_helper_utest.xml)
set_tests_properties(ut_libge_helper_utest PROPERTIES LABELS "ut;ut_dflow;ut_libge_helper_utest")