# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# test files
set(COMMON_TEST_FILES
    "graph/passes/graph_builder_utils.cc"
    "test.cc"
)

set(HETEROGENEOUS_MODEL_TEST_FILES
    "common/execution_runtime_unittest.cc"
    "common/model_relation_unittest.cc"
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
)

SET(DFLOW_EXECUTOR_TEST_FILES
        "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/executor/flow_model_manager_unittest.cc"
)

SET(DFLOW_SESSION_TEST_FILES
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_api_unittest.cc"
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_graph_manager_unittest.cc"
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_session_manager_unittest.cc"
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
)

target_include_directories(ut_libge_helper_utest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}/dflow
        ${AIR_CODE_DIR}/dflow/inc
        ${AIR_CODE_DIR}/dflow/inc/data_flow
        ${AIR_CODE_DIR}/dflow/deployer/
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/tests
        ${AIR_CODE_DIR}/tests/dflow/runner/ut
        ${AIR_CODE_DIR}/tests/dflow/runner/ut/ge
        ${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/stub
        ${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/heterogeneous/
        ${AIR_CODE_DIR}/tests/framework/ge_graph_dsl/include
        ${AIR_CODE_DIR}/tests/framework/ge_running_env/include
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${AIR_CODE_DIR}/tests/depends/runtime/src
        ${AIR_CODE_DIR}/tests/depends/ascendcl/src
        ${ASCEND_INSTALL_PATH}/include
        "${CMAKE_BINARY_DIR}/proto/data_flow_protos"
        "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto"
        "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto"
        "${CMAKE_BINARY_DIR}/proto/graphengine_protos"
        "${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto"
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
)

target_link_options(ut_libge_helper_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_libge_helper_utest
    intf_llt_pub
    metadef_headers
    ascendcl_stub
    datagw_headers
    ascend_hal_headers
    runtime_headers
    dflow_runner
    ge_compiler
    flow_graph
    mmpa_stub
    udf_compiler
    model_deployer
    deployer_daemon_static
    npu_executor_static
    npu_sched_model_loader
    data_flow_base
    ge_runner_v2
    ge_runner
    cpu_compiler
    -Wl,--whole-archive
    ge_runtime_stub
    ge_running_env
    -Wl,--no-whole-archive
    GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lpthread
    -Wl,--no-as-needed memset_stubs GTest::gmock -Wl,--as-needed
)

add_test(NAME ut_libge_helper_utest COMMAND ut_libge_helper_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_helper_utest.xml)
set_tests_properties(ut_libge_helper_utest PROPERTIES LABELS "ut;ut_dflow;ut_libge_helper_utest")