# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(ut_ascendcl_c_utest)
set(CMAKE_CXX_STANDARD 11)

set(UT_FILES
    "main.cc"
    "acl_mdl_desc_test.cc"
    "acl_mdl_test.cc"
)

set(UT_STUBS
    "${CMAKE_CURRENT_SOURCE_DIR}/gmock_executor/ge_executor_stub.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/stub/acl_ge_stub.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime_stub.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/stub/mmpa_api_stub.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/stub/model_config_rt.c"
)

set(SRC_FILES
    "${AIR_CODE_DIR}/api/acl_c/acl_model/model/model_config.c"
    "${AIR_CODE_DIR}/api/acl_c/acl_model/model/model_desc.c"
    "${AIR_CODE_DIR}/api/acl_c/acl_model/model/model_exec_config.c"
    "${AIR_CODE_DIR}/api/acl_c/acl_model/model/model.c"
    "${AIR_CODE_DIR}/api/acl_c/acl_model/types/tensor_desc_internal.c"
)

set(C_BASE_JSON_FILES
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/json_parser.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/mem_pool.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/ref_obj.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/sort_vector.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/vector.c
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:${AIR_CODE_DIR}/tests/test_c/c_base/src/error_manager.c>
)

############ ut_ascendcl_c_utest ############
add_executable(ut_ascendcl_c_utest
    ${C_BASE_JSON_FILES}
    ${UT_FILES}
    ${SRC_FILES}
    ${UT_STUBS}
)

#include directories
target_include_directories(ut_ascendcl_c_utest PRIVATE
    ${AIR_CODE_DIR}/api/acl_c/acl_model/model
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/acl/stub
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/acl/gmock_executor
)

target_compile_options(ut_ascendcl_c_utest PRIVATE
    -fsanitize=address -fsanitize=leak
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wall -Wextra -Wfloat-equal
    # -Werror
)

target_compile_definitions(ut_ascendcl_c_utest PRIVATE
    google=ascend_private
    ENABLE_DVPP_INTERFACE
    RUN_TEST
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>
)

target_link_libraries(ut_ascendcl_c_utest PRIVATE
    -fsanitize=address -fsanitize=leak
    $<BUILD_INTERFACE:intf_pub>
    msprof_headers
    runtime_headers
    c_sec
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    GTest::gmock_main
    -lrt
    -ldl
    -lpthread
    -lgcov
)
