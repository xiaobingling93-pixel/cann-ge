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
include_directories(${METADEF_DIR}/inc/common/util/trace_manager)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${GE_METADEF_DIR})
include_directories(${GE_METADEF_DIR}/exe_graph)
include_directories(${GE_METADEF_TEST_DIR}/depends)
include_directories(${GE_METADEF_DIR}/graph)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

file(GLOB_RECURSE UT_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/ut/expression/testcase/*.cc")
file(GLOB_RECURSE FAKER_SRCS CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/depends/faker/*.cc")
add_executable(ut_expression ${UT_FILES} ${FAKER_SRCS})

target_compile_options(ut_expression PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -D_GLIBCXX_USE_CXX11_ABI=0
    -fno-access-control
)

target_compile_definitions(ut_expression PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    FUNC_VISIBILITY
)

# intf_pub包含stdc++11 当前需要按照更高版本来编译ut
target_link_libraries(ut_expression PRIVATE
    -lgcov
    -Wl,--no-as-needed
    platform
    slog_headers
    metadef_headers
    register
    rt2_registry_static
    opp_registry
    slog
    symengine Boost::boost aihac_symbolizer
    GTest::gtest GTest::gtest_main ascend_protobuf error_manager graph_base graph slog_stub c_sec json mmpa_stub -lrt -ldl
)

target_include_directories(ut_expression PRIVATE
    ${GE_METADEF_TEST_DIR}/depends
    ${GE_METADEF_TEST_DIR}/ut/ascendc_ir
)

add_test(NAME ut_expression COMMAND ut_expression --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_expression.xml)
set_tests_properties(ut_expression PROPERTIES LABELS "ut;ge_rt;ut_expression")