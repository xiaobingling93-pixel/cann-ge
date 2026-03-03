# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${AIR_CODE_DIR}/inc)
include_directories(${AIR_CODE_DIR}/inc/external)
include_directories(${AIR_CODE_DIR}/tests/framework/macro_utils)

include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

set(UT_FILES
    "testcase/ge_graph/ge_anchor_utils_unittest.cc"
    "testcase/ge_graph/ge_def_type_unittest.cc"
    "testcase/ge_graph/ge_graph_anchor_unittest.cc"
    "testcase/ge_graph/ge_model_serialize_unittest.cc"
    "testcase/ge_graph/ge_node_unittest.cc"
    "testcase/ge_graph/ge_opdesc_unittest.cc"
    "testcase/ge_graph/ge_tensor_unittest.cc"
    "testcase/ge_graph/graph_builder_utils.cc"
    "testcase/ge_graph/ge_reg_unittest.cc"
    "testcase/ge_graph/ge_graph_dsl_check_test.cc"
    "testcase/ge_graph/ge_opsproto_manager_unittest.cc"
    "testcase/ge_graph/ge_operator_unittest.cc"
    "testcase/ge_graph/ge_model_unittest.cc"
)

add_executable(ut_libgraph ${UT_FILES} ${AIR_CODE_DIR}/base/common/types.cc)

target_compile_options(ut_libgraph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
)

target_compile_definitions(ut_libgraph PRIVATE
    google=ascend_private
    FUNC_VISIBILITY
)

target_link_libraries(ut_libgraph
    intf_pub
    metadef_headers
    GTest::gtest
    GTest::gtest_main
    ge_graph_dsl
)

add_test(NAME ut_libgraph COMMAND ut_libgraph --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libgraph.xml)
set_tests_properties(ut_libgraph PROPERTIES LABELS "ut;ge_common;ut_libgraph")