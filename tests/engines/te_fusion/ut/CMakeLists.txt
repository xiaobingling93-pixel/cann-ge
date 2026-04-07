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

protobuf_generate(tefusion_ut PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

add_executable(tefusion_ut
        ${PROTO_HDRS}
        ${TEFUSION_SRC_CC_LIST}
        ${CMAKE_CURRENT_LIST_DIR}/testcase/main.cc
        ${CMAKE_CURRENT_LIST_DIR}/stub/Python.cc
   	${CMAKE_CURRENT_LIST_DIR}/stub/Python_stub.cc
   	${CMAKE_CURRENT_LIST_DIR}/stub/te_llt_utils.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/fe_llt_utils.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/pass_manager.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/ub_pass_manager.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/graph_builder_utils.cc
        ${FE_TEST_ROOT_PATH}/graph_constructor/graph_constructor.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/stub/common_stub.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/stub/example_utest_stub.cc 
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/api/fusion_api_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/common/te_config_info_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/cache/te_cache_manager_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/python_adapter/py_object_utils_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/python_adapter/py_decouple_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/python_adapter/py_adapter_manager_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_base.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_test.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_fuzzy_build.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_build_file_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_py_obj_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_binary_ut.cc
 	${CMAKE_CURRENT_LIST_DIR}/testcase/op_compile_adapter/testcase/te_fusion_dfx_ut.cc)

target_include_directories(tefusion_ut PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${FE_ROOT_PATH}/utils
        ${FE_ROOT_PATH}/optimizer
        ${FE_ROOT_PATH}/fusion
        ${AIR_CODE_DIR}/tests/engines/nn_engine/graph_constructor/
        ${AIR_CODE_DIR}/compiler/opcompiler/inc
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends/ops_stub
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/graph_metadef/exe_graph
        ${GTEST_INCLUDE}
        ${CMAKE_BINARY_DIR}/proto/tefusion_ut
        ${AIR_CODE_DIR}/compiler/opcompiler/op_compile_adapter/source
  	${AIR_CODE_DIR}/tests/engines/te_fusion/ut/stub
  	${AIR_CODE_DIR}/tests/engines/te_fusion/ut/testcase/op_compile_adapter/stub
        ${TEFUSION_TEST_ROOT_PATH})

target_compile_options(tefusion_ut PRIVATE
        $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -g>
        -D__OPTIMIZER_KT_TEST__
        -DDAVINCI_CLOUD
        -Dgoogle=ascend_private)

target_compile_definitions(tefusion_ut PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>)

target_link_directories(tefusion_ut PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
    ${ASCEND_INSTALL_PATH}/devlib
)

target_link_libraries(tefusion_ut PRIVATE
        $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
        air_headers
        metadef_headers
        slog_headers
        mmpa_headers
        cce_headers
        msprof_headers
        runtime_headers
        op_compile_adapter_headers
        -Wl,--no-as-needed
        -lrt -ldl -lpthread -lgcov
        $<BUILD_INTERFACE:atrace_headers>
        atrace_stub
        exe_graph
        lowering
        graph
        metadef
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
        c_sec_static
        mmpa_stub
        slog_stub
        unified_dlog
        cann_kb_stub
        easy_graph
        ge_graph_dsl
        intf_llt_pub
        aicore_utils
        fe
        fusion_pass
        acl_rt
        graph_base
        ge_common_base
   	GTest::gmock
   	mockcpp_static)
