# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
# include directories
include_directories(${CMAKE_SOURCE_DIR})
add_compile_definitions(TOP_DIR=\"${CMAKE_SOURCE_DIR}\")
file(GLOB_RECURSE DFLOW_UT_FILES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/dflow/flow_graph/ut/dflow/flow_graph/**.cc"
        ${DFLOW_CODE_DIR}/flow_graph/**.cc
)
message("DFLOW_UT_FILES=${DFLOW_UT_FILES}")
add_executable(ut_flow_graph ${DFLOW_UT_FILES})
target_include_directories(ut_flow_graph PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${DFLOW_CODE_DIR}
        ${DFLOW_CODE_DIR}/inc
        ${DFLOW_CODE_DIR}/compiler
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/ge
        ${CMAKE_BINARY_DIR}/proto/data_flow_protos
)

target_compile_options(ut_flow_graph PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -D_GLIBCXX_USE_CXX11_ABI=0
)

target_compile_definitions(ut_flow_graph PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
        FUNC_VISIBILITY
)

target_link_options(ut_flow_graph PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_flow_graph PRIVATE
        intf_pub
        -lgcov
        metadef_headers
        ascend_protobuf
        c_sec_headers
        c_sec
        slog_stub
        json
        graph
        graph_base
        flow_graph_protos_obj
        -Wl,--no-as-needed
        unified_dlog
        GTest::gtest GTest::gtest_main -lrt -ldl
)

add_test(NAME ut_flow_graph COMMAND ut_flow_graph --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_flow_graph.xml)
set_tests_properties(ut_flow_graph PROPERTIES LABELS "ut;ut_dflow;ut_flow_graph")