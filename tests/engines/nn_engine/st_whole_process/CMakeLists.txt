# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(fe_st_whole_process)
file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/om.proto"
        "${METADEF_PROTO_DIR}/insert_op.proto"
        "${METADEF_PROTO_DIR}/var_manager.proto"
        "${METADEF_PROTO_DIR}/flow_model.proto"
        "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
        "${METADEF_PROTO_DIR}/task_mobile.proto")

protobuf_generate(fe_st_whole_process PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# compile for fe_st_whole_process
add_executable(fe_st_whole_process
        ${PROTO_HDRS}
        ${FE_SRC_CC_LIST}
        ${FFTS_SRC_CC_LIST}
        ${CMAKE_CURRENT_LIST_DIR}/testcase/main.cc
        ${FE_TEST_ROOT_PATH}/framework/fe_running_env/src/env/fe_running_env.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/graph_constructor.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/fe_llt_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/stub/te_fusion_api.cc
        ${CMAKE_CURRENT_LIST_DIR}/stub/aicore_ops_kernel_builder_stub.cc
        #${CMAKE_CURRENT_LIST_DIR}/testcase/fe_whole_process/test_fe_whole_process_st_310p3.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/fe_whole_process/test_fe_whole_process_st_910a.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/fe_whole_process/test_fe_whole_process_st_910b.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/fe_whole_process/test_fe_whole_process_st_task_check.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/fe_whole_process/test_fe_whole_process_st_310b.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/fe_whole_process/test_fe_whole_process_st_nano.cc
        )

target_include_directories(fe_st_whole_process PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${FE_ROOT_PATH}/executor
        ${FE_ROOT_PATH}/inc
        ${FE_ROOT_PATH}/optimizer
        ${FE_ROOT_PATH}/utils
        ${FE_ROOT_PATH}/fusion
        ${FE_ROOT_PATH}/opskernel

        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/graph/
        ${METADEF_DIR}/inc/common
        ${METADEF_DIR}/inc/exe_graph/
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/inc/register
        ${METADEF_DIR}/third_party/transformer
        ${METADEF_DIR}/inc/common/util
        ${METADEF_DIR}

        #ge_compiler_depends
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/base

        ${AIR_CODE_DIR}/compiler/engines/ffts_engine
        ${AIR_CODE_DIR}/compiler/engines/ffts_engine/inc
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends/ops_stub
        ${AIR_CODE_DIR}/tests/engines/nn_engine/graph_constructor
        ${ASCEND_INSTALL_PATH}/include/experiment/opat
        ${GTEST_INCLUDE}
        ${CMAKE_BINARY_DIR}/proto/fe_st_whole_process
        ${FE_TEST_ROOT_PATH}
        )

target_compile_options(fe_st_whole_process PRIVATE
        -D__OPTIMIZER_KT_TEST__
        -DDAVINCI_CLOUD
        -g
        -Dgoogle=ascend_private)

target_compile_definitions(fe_st_whole_process PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>)

target_link_directories(fe_st_whole_process PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(fe_st_whole_process PRIVATE
        slog_headers
        mmpa_headers
        msprof_headers
        runtime_headers
        opcompiler_headers
        -Wl,--no-as-needed
        -lrt -ldl -lpthread -lgcov
        $<BUILD_INTERFACE:atrace_headers>
        atrace_stub
        exe_graph
        lowering
        graph
        register
        opp_registry
        ge_runner_pkg
        ge_common_pkg
        ge_compiler_pkg
        runtime_stub_fe
        ascend_protobuf
        c_sec
        platform
        compress
        error_manager
        json
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
        c_sec_static
        easy_graph
        #cann_kb
        ge_graph_dsl
        mmpa_stub
        slog_stub
        fe_env_mocker
        intf_llt_pub)
