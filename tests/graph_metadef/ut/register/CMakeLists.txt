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
include_directories(${AIR_CODE_DIR}/tests)
include_directories(${AIR_CODE_DIR}/tests/graph_metadef/ut)
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${GE_METADEF_DIR})
include_directories(${GE_METADEF_DIR}/register)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include)
include_directories(${AIR_CODE_DIR}/inc/framework)

file(GLOB_RECURSE REGISTER_UT_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/ut/register/*.cc" )
file(GLOB_RECURSE UTILS_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/ut/graph/common/*.cc" )

add_executable(ut_register
    ${REGISTER_UT_FILES} ${UTILS_FILES}
)
add_compile_definitions(CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\")
target_compile_options(ut_register PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
    -fvisibility=hidden
)

target_compile_definitions(ut_register PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_link_options(ut_register PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ut_register
    intf_pub
    metadef_headers
    runtime_headers
    slog_headers
    msprof_headers
    exe_graph lowering register graph graph_base mmpa
    opp_registry
    GTest::gtest
    GTest::gtest_main
    ascend_protobuf
    slog_stub
    platform
    json
    c_sec
    error_manager
    -lrt
    -ldl
    -lgcov
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
    ge_runtime_stub
)

target_include_directories(ut_register PRIVATE
    ${GE_METADEF_TEST_DIR}/ut/graph/common
    ${AIR_CODE_DIR}/inc/graph_metadef/external
)

add_test(NAME ut_register COMMAND ut_register --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_register.xml)
set_tests_properties(ut_register PROPERTIES LABELS "ut;ge_rt;ut_register")