# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_directories(${AIR_CODE_DIR}/tests/)

MESSAGE("copy st run data from ${AIR_CODE_DIR}/tests/dflow/runner/st/st_run_data to ${CMAKE_CURRENT_BINARY_DIR}/")
file(COPY ${AIR_CODE_DIR}/tests/dflow/runner/st/st_run_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

################################################################################
# build helper_runtime_test
add_executable(helper_runtime_test
    test_main.cc
    build_cache_system_test.cc
    test_heterogeneous_runtime.cc
    test_helper_runtime.cc
    test_helper_planner.cc
    test_deployer_daemon.cc
    test_memory_statistic.cc
    data_flow_graph/data_flow_graph_test.cc
    data_flow_graph/data_flow_graph_batch_attr_test.cc
    data_flow_graph/inner_pp_test.cc
    data_flow_runtime/test_data_flow_runtime.cc
    data_flow_runtime/test_npu_sched_model.cc
    data_flow_session/test_data_flow_api.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/dgw_client_st_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/tsd_client_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/caas_dataflow_auth_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/re_mem_group_st_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/process_utils_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/deployer_client_stub.cc
)

target_compile_definitions(helper_runtime_test PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
)

target_include_directories(helper_runtime_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/dflow/deployer
    ${CMAKE_BINARY_DIR}/proto/deployer_proto/
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/dflow/base
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto
    ${AIR_CODE_DIR}/dflow
    ${AIR_CODE_DIR}/dflow/inc
    ${AIR_CODE_DIR}/dflow/inc/data_flow
)

target_link_options(helper_runtime_test PRIVATE
    -Wl,-Bsymbolic
)

target_compile_options(helper_runtime_test PRIVATE
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
    -Wno-memset-elt-size
)

target_link_libraries(helper_runtime_test
    intf_llt_pub
    -Wl,-z,muldefs
    model_deployer
    dflow_runner
    ge_compiler
    flow_graph
    ge_runner_v2
    ge_runner
    npu_sched_model_loader
    deployer_daemon_static
    npu_executor_static
    udf_compiler
    cpu_compiler
    ge_executor
    hybrid_executor
    davinci_executor
    mmpa
    -Wl,--whole-archive
    df_memset_stubs
    -Wl,--no-whole-archive
    ge_graph_dsl df_st_stubs ge_running_env ge_runtime_stub ascendcl_stub
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    data_flow_base
    -lrt -ldl -lpthread
)

include(CTest)
enable_testing()
add_test(NAME test COMMAND helper_runtime_test)

add_test(NAME helper_runtime_test COMMAND helper_runtime_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/helper_runtime_test.xml)
set_tests_properties(helper_runtime_test PROPERTIES LABELS "st;st_dflow;helper_runtime_test")
