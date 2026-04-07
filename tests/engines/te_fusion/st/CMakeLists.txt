# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/om.proto"
        "${METADEF_PROTO_DIR}/insert_op.proto"
        "${METADEF_PROTO_DIR}/flow_model.proto"
        "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
        "${METADEF_PROTO_DIR}/task_mobile.proto")

protobuf_generate(tefusion_st PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

add_executable(tefusion_st
        ${PROTO_HDRS}
        ${TEFUSION_SRC_CC_LIST}
        ${CMAKE_CURRENT_LIST_DIR}/stub/Python.cc
        ${CMAKE_CURRENT_LIST_DIR}/stub/Python_stub.cc
        ${CMAKE_CURRENT_LIST_DIR}/stub/te_llt_utils.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/fe_llt_utils.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/pass_manager.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/ub_pass_manager.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/graph_builder_utils.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/graph_constructor.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/stub/common_stub.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/stub/example_stest_stub.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/api/fusion_api_st.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/cache/te_cache_manager_st.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/common/te_config_info_st.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_base.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_binary_st.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_build_file_st.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_dfx_st.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_fuzzy_build.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_py_obj_st.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_test.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/main.cc)

target_include_directories(tefusion_st PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${FE_ROOT_PATH}/utils
        ${FE_ROOT_PATH}/optimizer
        ${FE_ROOT_PATH}/fusion
        ${ASCEND_INSTALL_PATH}/include
        ${AIR_CODE_DIR}/tests/engines/nn_engine/graph_constructor
        ${AIR_CODE_DIR}/compiler/opcompiler/op_compile_adapter/source
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/graph_metadef/exe_graph
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/compiler/opcompiler/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends/ops_stub
        ${GTEST_INCLUDE}
        ${CMAKE_BINARY_DIR}/proto/tefusion_st
        ${TEFUSION_TEST_ROOT_PATH}
        ${AIR_CODE_DIR}/compiler/opcompiler/op_compile_adapter/source
        ${AIR_CODE_DIR}/tests/engines/te_fusion/st/stub
        ${AIR_CODE_DIR}/tests/engines/te_fusion/st/testcase/op_compile_adapter/stub)

target_compile_options(tefusion_st PRIVATE
        -D__OPTIMIZER_KT_TEST__
        -DDAVINCI_CLOUD
        -Dgoogle=ascend_private)

target_compile_definitions(tefusion_st PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>)

target_link_directories(tefusion_st PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
    ${ASCEND_INSTALL_PATH}/devlib
)

target_link_libraries(tefusion_st PRIVATE
        msprof_headers
        cce_headers
        runtime_headers
        op_compile_adapter_headers
        -Wl,--no-as-needed
        -lrt -ldl -lpthread -lgcov
        exe_graph
        lowering
        graph
        air_headers
        metadef_headers
        metadef
        $<BUILD_INTERFACE:atrace_headers>
        atrace_stub
        register
        opp_registry
        op_tiling_rt2_stub
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
        mockcpp_static
        c_sec_static
        easy_graph
        ge_graph_dsl
        mmpa_stub
        op_compile_adapter_headers
        slog_stub
        unified_dlog
        cann_kb_stub
        fe
        aicore_utils
        fusion_pass
        intf_llt_pub
        ge_common_base
        acl_rt)