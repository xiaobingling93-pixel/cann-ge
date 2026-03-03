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
include_directories(${AIR_CODE_DIR}/tests/ge/ut)
include_directories(${AIR_CODE_DIR}/tests/ge/ut/ge)

MESSAGE("copy st run data from ${AIR_CODE_DIR}/tests/dflow/runner/st/st_run_data to ${CMAKE_CURRENT_BINARY_DIR}/")
file(COPY ${AIR_CODE_DIR}/tests/dflow/runner/st/st_run_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

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
        "${METADEF_PROTO_DIR}/attr_group_base.proto"
        "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
        "${METADEF_PROTO_DIR}/task_mobile.proto"
        )

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

################################################################################
# build helper_runtime_test
add_executable(helper_runtime_test
    test_main.cc
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
)

target_compile_definitions(helper_runtime_test PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
)

target_include_directories(helper_runtime_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/dflow/deployer
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/runtime/deploy/deployer/device_mgr
    ${AIR_CODE_DIR}/runtime/deploy/abnormal_status_handler
    ${CMAKE_BINARY_DIR}/proto/ge
    ${CMAKE_BINARY_DIR}/proto/ge/proto
    ${ASCGEN_DIR}/inc
    ${AIR_CODE_DIR}/inc
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${AIR_CODE_DIR}/dflow/base
    ${AIR_CODE_DIR}/dflow/base/exec_runtime
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto
    ${AIR_CODE_DIR}/dflow
    ${AIR_CODE_DIR}/dflow/inc
    ${AIR_CODE_DIR}/dflow/inc/data_flow
)

target_compile_options(helper_runtime_test PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -O0 -g
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
)

target_link_libraries(helper_runtime_test
    intf_pub
    graphengine helper_runtime
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
    ge_graph_dsl df_st_stubs ge_running_env ge_runtime_stub ascendcl_stub
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    data_flow_base
)

include(CTest)
enable_testing()
add_test(NAME test COMMAND helper_runtime_test)

add_test(NAME helper_runtime_test COMMAND helper_runtime_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/helper_runtime_test.xml)
set_tests_properties(helper_runtime_test PROPERTIES LABELS "st;st_dflow;helper_runtime_test")
